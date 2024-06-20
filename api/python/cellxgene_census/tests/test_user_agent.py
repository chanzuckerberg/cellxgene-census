import json
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import proxy
import pytest
import requests

import cellxgene_census

if TYPE_CHECKING:
    pass


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
        "cellxgene_census._testing.logger_proxy.RequestLoggerPlugin",
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


@pytest.fixture
def test_specific_useragent() -> str:
    """Sets custom user agent addendum for every test so they can be uniqueley identified."""
    current_test_name = os.environ["PYTEST_CURRENT_TEST"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CELLXGENE_CENSUS_USERAGENT", current_test_name)
        yield current_test_name


@pytest.fixture()
def proxy_instance(proxy_server):
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

    def _check_proxy():
        # For each new log written by the test, check that the correct headers were written
        with proxy_server.logpth.open("r") as f:
            records = [json.loads(line) for line in f.readlines()]
        records = records[prev_lines:]
        return records
        # assert len(records) > 0
        # for record in records:
        #     if record["method"] == "CONNECT":
        #         continue
        #     headers = record["headers"]
        #     user_agent = headers["user-agent"]
        #     assert "cellxgene-census-python" in user_agent
        #     assert test_specific_useragent in user_agent
        #     assert "foo" in user_agent

    # Run test
    yield _check_proxy

    # with proxy_server.logpth.open("r") as f:
    #     records = [json.loads(line) for line in f.readlines()]
    # records = records[prev_lines:]
    # assert len(records) > 0
    # for record in records:
    #     if record["method"] == "CONNECT":
    #         continue
    #     headers = record["headers"]
    #     user_agent = headers["user-agent"]
    #     assert "cellxgene-census-python" in user_agent
    #     assert test_specific_useragent in user_agent


def check_proxy_records(records: list[dict], custom_user_agent: None | str = None) -> None:
    n_records = 0
    for record in records:
        if record["method"] == "CONNECT":
            continue
        n_records += 1
        headers = record["headers"]
        user_agent = headers["user-agent"]
        assert "cellxgene-census-python" in user_agent
        if custom_user_agent:
            assert custom_user_agent in user_agent
    assert n_records > 0, "No requests were intercepted"


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


@pytest.fixture(scope="session")
def small_dataset_id() -> str:
    # TODO: REMOVE, copied from test_open
    with cellxgene_census.open_soma(census_version="latest") as census:
        census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()

    small_dataset = census_datasets.nsmallest(1, "dataset_total_cell_count").iloc[0]
    assert isinstance(small_dataset.dataset_id, str)
    return small_dataset.dataset_id


def test_download_w_proxy_fixture(small_dataset_id, proxy_instance, tmp_path, test_specific_useragent):
    # Use of proxy_instance forces test to use a proxy and will check headers of requests made via that proxy
    adata_path = tmp_path / "adata.h5ad"
    cellxgene_census.download_source_h5ad(small_dataset_id, adata_path.as_posix(), census_version="latest")

    records = proxy_instance()
    check_proxy_records(records, custom_user_agent=test_specific_useragent)


def test_query_w_proxy_fixture(proxy_instance):
    with cellxgene_census.open_soma(census_version="stable") as census:
        _ = cellxgene_census.get_obs(census, "Mus musculus", coords=slice(100, 300))

    records = proxy_instance()
    print(records)
    check_proxy_records(records)
    assert False
