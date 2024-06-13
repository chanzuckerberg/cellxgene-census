import json
from pathlib import Path

import proxy
import pytest
import tiledbsoma as soma

TEST_MARKERS_SKIPPED_BY_DEFAULT = ["expensive", "experimental"]


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


_BASE_CERT_DIR = "/home/ubuntu/github/proxy.py"


# Fixtures for census objects
@pytest.fixture(scope="session")
def proxied_soma_context() -> "soma.options.SOMATileDBContext":
    return _proxied_soma_context(8899)


def _proxied_soma_context(port: int) -> "soma.options.SOMATileDBContext":
    from cellxgene_census._open import DEFAULT_TILEDB_CONFIGURATION, get_default_soma_context

    # Directing request through proxy so we can check the headers
    tiledb_config = DEFAULT_TILEDB_CONFIGURATION.copy()
    # tiledb_config["vfs.s3.scheme"] = "http"
    tiledb_config["vfs.s3.proxy_host"] = "localhost"
    # tiledb_config["vfs.s3.proxy_scheme"] = "http"
    tiledb_config["vfs.s3.proxy_port"] = str(port)
    # tiledb_config["vfs.s3.ca_file"] = f"{_BASE_CERT_DIR}/ca-cert.pem"
    tiledb_config["vfs.s3.verify_ssl"] = "false"

    return get_default_soma_context(tiledb_config)


# @pytest.fixture()
# def http_urls():
#     import cellxgene_census._release_directory, cellxgene_census.experimental._embedding

#     with pytest.MonkeyPatch.context() as mp:
#         mp.setattr(cellxgene_census._release_directory, "CELL_CENSUS_RELEASE_DIRECTORY_URL", "http://census.cellxgene.cziscience.com/cellxgene-census/v1/release.json")
#         mp.setattr(cellxgene_census._release_directory, "CELL_CENSUS_MIRRORS_DIRECTORY_URL", "http://census.cellxgene.cziscience.com/cellxgene-census/v1/mirrors.json")
#         mp.setattr(cellxgene_census.experimental._embedding, "CELL_CENSUS_EMBEDDINGS_MANIFEST_URL", "http://contrib.cellxgene.cziscience.com/contrib/cell-census/contributions.json")
#         mp.setenv("HTTP_PROXY", "http://localhost:8899")
#         yield


@pytest.fixture(scope="session")
def census() -> soma.Collection:
    import cellxgene_census

    return cellxgene_census.open_soma(census_version="latest")


@pytest.fixture(scope="session")
def lts_census() -> soma.Collection:
    import cellxgene_census

    return cellxgene_census.open_soma(census_version="stable")


# @pytest.fixture():
# def test_request_context():
#     from unittest.mock import patch
#     with patch("cellxgene_census.")


class ProxyInstance:
    def __init__(self, proxy_obj: proxy.Proxy, logpth: Path, soma_context):
        self.proxy = proxy_obj
        self.logpth = logpth
        self.soma_context = soma_context

    @property
    def port(self) -> int:
        return self.proxy.flags.port


@pytest.fixture
def proxy_server(tmp_path: Path):
    # Set up
    from proxy.plugin import CacheResponsesPlugin

    import cellxgene_census._release_directory
    import cellxgene_census.experimental._embedding

    logpth = tmp_path / "proxy.log"

    # Copied from TestCase setup from proxy.py: https://github.com/abhinavsingh/proxy.py/blob/develop/proxy/testing/test_case.py#L23

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
        # Not sure why these are neccesary, but I can't see the tiledb headers without these lines
        "--ca-key-file",
        f"{_BASE_CERT_DIR}/ca-key.pem",
        "--ca-cert-file",
        f"{_BASE_CERT_DIR}/ca-cert.pem",
        "--ca-signing-key-file",
        f"{_BASE_CERT_DIR}/ca-signing-key.pem",
        # Proxy.py doesn't seem to be generating any logs, so I am writing all these via the ProxyPlugin
        "--log-file",
        str(logpth),
        # "--log-level", "DEBUG",
    ]
    proxy_obj = proxy.Proxy(PROXY_PY_STARTUP_FLAGS)
    proxy_obj.flags.plugins[b"HttpProxyBasePlugin"].append(
        CacheResponsesPlugin,
    )
    with proxy_obj:
        assert proxy_obj.acceptors
        proxy.TestCase.wait_for_server(proxy_obj.flags.port)
        # Now that proxy is set up, set relevant environment variables/ constants
        with pytest.MonkeyPatch.context() as mp:
            # mp.setattr(
            #     cellxgene_census._release_directory,
            #     "CELL_CENSUS_RELEASE_DIRECTORY_URL",
            #     "http://census.cellxgene.cziscience.com/cellxgene-census/v1/release.json",
            # )
            # mp.setattr(
            #     cellxgene_census._release_directory,
            #     "CELL_CENSUS_MIRRORS_DIRECTORY_URL",
            #     "http://census.cellxgene.cziscience.com/cellxgene-census/v1/mirrors.json",
            # )
            # mp.setattr(
            #     cellxgene_census.experimental._embedding,
            #     "CELL_CENSUS_EMBEDDINGS_MANIFEST_URL",
            #     "http://contrib.cellxgene.cziscience.com/contrib/cell-census/contributions.json",
            # )
            mp.setenv("HTTP_PROXY", f"http://localhost:{proxy_obj.flags.port}")
            mp.setenv("HTTPS_PROXY", f"http://localhost:{proxy_obj.flags.port}")
            yield ProxyInstance(proxy_obj, logpth, soma_context=_proxied_soma_context(proxy_obj.flags.port))

    # Validate results in cleanup
    with logpth.open("r") as f:
        records = [json.loads(line) for line in f.readlines()]
    assert len(records) > 0
    for record in records:
        if record["method"] == "CONNECT":
            # TODO: IDK why this is happening, figure out later
            continue
        headers = record["headers"]
        user_agent = headers["user-agent"]
        assert "cellxgene-census-python" in user_agent
    # pylint: disable=unnecessary-dunder-call
