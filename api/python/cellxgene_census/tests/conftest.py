import json
import multiprocessing
import os
from functools import partial
from pathlib import Path

import proxy
import pytest
import requests
import tiledbsoma as soma

TEST_MARKERS_SKIPPED_BY_DEFAULT = ["expensive", "experimental"]

# tiledb will complain if this isn't set and a process is spawned. May cause segfaults on the proxy test if this isn't set.
multiprocessing.set_start_method("spawn", force=True)


def pytest_addoption(parser: pytest.Parser) -> None:
    for test_option in TEST_MARKERS_SKIPPED_BY_DEFAULT:
        parser.addoption(
            f"--{test_option}",
            action="store_true",
            dest=test_option,
            default=False,
            help=f"enable '{test_option}' decorated tests",
        )

    # Add option to set the census_version (not set by default)
    parser.addoption("--census_version", action="store", default="stable")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """This is called for every test"""

    # Configure census_version if used
    census_version = metafunc.config.option.census_version
    if "census_version" in metafunc.fixturenames:
        metafunc.parametrize("census_version", [census_version])


def pytest_configure(config: pytest.Config) -> None:
    """
    Exclude tests marked with any of the TEST_MARKERS_SKIPPED_BY_DEFAULT values, unless the corresponding explicit
    flag is specified by the user.
    """
    excluded_markexprs = []

    for test_option in TEST_MARKERS_SKIPPED_BY_DEFAULT:
        if not vars(config.option).get(test_option, False):
            excluded_markexprs.append(test_option)

    if config.option.markexpr and excluded_markexprs:
        config.option.markexpr += " and "
    config.option.markexpr += " and ".join([f"not {m}" for m in excluded_markexprs])


@pytest.fixture
def small_mem_context() -> soma.SOMATileDBContext:
    """used to keep memory usage smaller for GHA runners."""
    from cellxgene_census import get_default_soma_context

    return get_default_soma_context(tiledb_config={"soma.init_buffer_bytes": 32 * 1024**2})


@pytest.fixture(scope="session")
def census() -> soma.Collection:
    import cellxgene_census

    return cellxgene_census.open_soma(census_version="latest")


@pytest.fixture(scope="session")
def lts_census() -> soma.Collection:
    import cellxgene_census

    return cellxgene_census.open_soma(census_version="stable")


class ProxyInstance:
    def __init__(self, proxy_obj: proxy.Proxy, logpth: Path):
        self.proxy = proxy_obj
        self.logpth = logpth

    @property
    def port(self) -> int:
        return self.proxy.flags.port


@pytest.fixture(scope="session")
def proxy_server(
    tmp_path_factory: Path,
    ca_certificates: tuple[Path, Path, Path],
):
    from proxy.plugin import CacheResponsesPlugin

    import cellxgene_census

    tmp_path = tmp_path_factory.mktemp("proxy_logs")
    logpth = tmp_path / "proxy.log"
    key_file, cert_file, signing_keyfile = ca_certificates
    assert all(p.is_file() for p in (key_file, cert_file, signing_keyfile))

    # Adapted from TestCase setup from proxy.py: https://github.com/abhinavsingh/proxy.py/blob/develop/proxy/testing/test_case.py#L23
    PROXY_PY_STARTUP_FLAGS = [
        "--num-workers",
        "1",
        "--num-acceptors",
        "1",
        "--hostname",
        "127.0.0.1",
        "--port",
        "0",
        "--plugin",
        "cellxgene_census._testing.ProxyPlugin",
        "--ca-key-file",
        str(key_file),
        "--ca-cert-file",
        str(cert_file),
        "--request-logfile",
        str(logpth),
    ]
    proxy_obj = proxy.Proxy(PROXY_PY_STARTUP_FLAGS)
    proxy_obj.flags.plugins[b"HttpProxyBasePlugin"].append(
        CacheResponsesPlugin,
    )
    with proxy_obj:
        assert proxy_obj.acceptors
        proxy.TestCase.wait_for_server(proxy_obj.flags.port)
        proxy_instance = ProxyInstance(proxy_obj, logpth)

        # Now that proxy is set up, set relevant environment variables/ constants to make all request making libraries use proxy
        with pytest.MonkeyPatch.context() as mp:
            # Both requests and s3fs use these environment variables:
            mp.setenv("HTTP_PROXY", f"http://localhost:{proxy_obj.flags.port}")
            mp.setenv("HTTPS_PROXY", f"http://localhost:{proxy_obj.flags.port}")

            # s3fs
            mp.setattr(
                cellxgene_census._open,
                "DEFAULT_S3FS_KWARGS",
                {
                    "anon": True,
                    "cache_regions": True,
                    "use_ssl": False,  # So we can inspect the requests on the proxy
                },
            )

            # Requests
            mp.setattr(requests, "get", partial(requests.request, "get", verify=False))

            # Tiledb
            tiledb_config = cellxgene_census._open.DEFAULT_TILEDB_CONFIGURATION.copy()
            tiledb_config["vfs.s3.proxy_host"] = "localhost"
            tiledb_config["vfs.s3.proxy_port"] = str(proxy_instance.port)
            tiledb_config["vfs.s3.verify_ssl"] = "false"
            mp.setattr(
                cellxgene_census._open,
                "DEFAULT_TILEDB_CONFIGURATION",
                tiledb_config,
            )

            yield proxy_instance


@pytest.fixture()
def proxy_instance(proxy_server, test_specific_useragent):
    """Test specific fixture exposing the proxy server.

    While a proxy server is started for every test session, this fixture
    captures only the output that is written for a specific tests. These
    logged requests are checked to make sure have the correct headers.
    """
    # If logs have already been written, count how many
    if proxy_server.logpth.is_file():
        with proxy_server.logpth.open("r") as f:
            prev_lines = len(f.readlines())
    else:
        prev_lines = 0

    # Run test
    yield proxy_server

    # For each new log written by the test, check that the correct headers were written
    with proxy_server.logpth.open("r") as f:
        records = [json.loads(line) for line in f.readlines()]
    records = records[prev_lines:]
    assert len(records) > 0
    for record in records:
        if record["method"] == "CONNECT":
            continue
        headers = record["headers"]
        user_agent = headers["user-agent"]
        assert "cellxgene-census-python" in user_agent
        assert test_specific_useragent in user_agent


@pytest.fixture
def test_specific_useragent() -> str:
    """Sets custom user agent addendum for every test so they can be uniqueley identified."""
    current_test_name = os.environ["PYTEST_CURRENT_TEST"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CELLXGENE_CENSUS_USERAGENT", current_test_name)
        yield current_test_name


@pytest.fixture(scope="session")
def ca_certificates(tmp_path_factory) -> tuple[Path, Path, Path]:
    # Adapted from https://github.com/abhinavsingh/proxy.py/blob/a7077cf8db3bb66a6667a9d968a401e8f805e092/Makefile#L68C1-L82C49
    # TODO: Figure out if we can remove this. Currently seems neccesary for intercepting tiledb s3 requests
    cert_dir = tmp_path_factory.mktemp("ca-certificates")
    KEY_FILE = cert_dir / "ca-key.pem"
    CERT_FILE = cert_dir / "ca-cert.pem"
    SIGNING_KEY_FILE = cert_dir / "ca-signing-key.pem"
    assert proxy.common.pki.gen_private_key(key_path=KEY_FILE, password="proxy.py")
    assert proxy.common.pki.remove_passphrase(key_in_path=KEY_FILE, password="proxy.py", key_out_path=KEY_FILE)
    assert proxy.common.pki.gen_public_key(
        public_key_path=CERT_FILE, private_key_path=KEY_FILE, private_key_password="proxy.py", subject="/CN=localhost"
    )
    assert proxy.common.pki.gen_private_key(key_path=SIGNING_KEY_FILE, password="proxy.py")
    assert proxy.common.pki.remove_passphrase(
        key_in_path=SIGNING_KEY_FILE, password="proxy.py", key_out_path=SIGNING_KEY_FILE
    )
    return (KEY_FILE, CERT_FILE, SIGNING_KEY_FILE)
