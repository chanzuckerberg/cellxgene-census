# mypy: ignore-errors
from __future__ import annotations

import json
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import proxy
import pytest
import requests
from urllib3.exceptions import InsecureRequestWarning

if TYPE_CHECKING:
    from _pytest.tmpdir import TempPathFactory

import cellxgene_census

# We are forcing the requests to be insecure so we can intercept them.
pytestmark = pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")


class ProxyInstance:
    def __init__(self, proxy_obj: proxy.Proxy, logpth: Path):
        self.proxy = proxy_obj
        self.logpth = logpth

    @property
    def port(self) -> int:
        return self.proxy.flags.port


@pytest.fixture(scope="session")
def ca_certificates(tmp_path_factory: TempPathFactory) -> tuple[Path, Path, Path]:
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
def proxy_server(
    tmp_path_factory: TempPathFactory,
    ca_certificates: tuple[Path, Path, Path],
):
    import cellxgene_census

    tmp_path = tmp_path_factory.mktemp("proxy_logs")
    # proxy.py can override passed ca-key-file and ca-cert-file with cached ones. So we create a fresh cache for each proxy server
    cert_cache_dir = tmp_path_factory.mktemp("certificates_cache")
    proxy_log_file = tmp_path / "proxy.log"
    request_log_file = tmp_path / "proxy_requests.log"
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
        "--ca-signing-key-file",
        str(signing_keyfile),
        "--ca-cert-dir",
        str(cert_cache_dir),
        "--log-file",
        str(proxy_log_file),
        "--request-log-file",
        str(request_log_file),
    ]
    proxy_obj = proxy.Proxy(PROXY_PY_STARTUP_FLAGS)
    with proxy_obj:
        assert proxy_obj.acceptors
        proxy.TestCase.wait_for_server(proxy_obj.flags.port)
        proxy_instance = ProxyInstance(proxy_obj, request_log_file)

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

            # requests
            mp.setattr(requests, "get", partial(requests.request, "get", verify=False))

            # tiledb
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
def collect_proxy_requests(proxy_server: ProxyInstance):
    """Test specific fixture exposing the proxy server.

    While a proxy server is started for every test session, this fixture
    captures only the output that is written for a specific tests. These
    logged requests can be checked to make sure have the correct headers.
    """
    # If logs have already been written, count how many
    if proxy_server.logpth.is_file():
        with proxy_server.logpth.open("r") as f:
            prev_lines = len(f.readlines())
    else:
        prev_lines = 0

    def _proxy_requests():
        # For each new log written by the test, check that the correct headers were written
        with proxy_server.logpth.open("r") as f:
            records = [json.loads(line) for line in f.readlines()]
        records = records[prev_lines:]
        return records

    # Run test
    yield _proxy_requests


@pytest.fixture(scope="session")
def small_dataset_id() -> str:
    with cellxgene_census.open_soma(census_version="latest") as census:
        census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()

    small_dataset = census_datasets.nsmallest(1, "dataset_total_cell_count").iloc[0]
    assert isinstance(small_dataset.dataset_id, str)
    return small_dataset.dataset_id


def check_proxy_records(records: list[dict], *, custom_user_agent: None | str = None, min_records: int = 1) -> None:
    # Check that there aren't two CONNECT requests in a row
    prev_was_connect = False
    for record in records:
        was_connect = record["method"] == "CONNECT"
        if prev_was_connect and was_connect:
            raise AssertionError(
                "Recieved multiple connect requests in a row. Some calls aren't being intercepted by the proxy."
            )

    # Check that headers were set correctly on intercepted requests
    n_records = 0
    for record in records:
        if record["method"] == "CONNECT":
            continue
        n_records += 1
        headers = record["headers"]
        user_agent = headers["user-agent"]
        assert "cellxgene-census-python" in user_agent
        assert cellxgene_census.__version__ in user_agent
        if custom_user_agent:
            assert custom_user_agent in user_agent
    assert n_records >= min_records, f"Fewer than min_records ({min_records}) were found."


def test_proxy_fixture(collect_proxy_requests: Callable[[], list[dict]]):
    """Test that our proxy testing setup is working as expected."""
    # Should just be downloading a json
    with pytest.warns(InsecureRequestWarning):
        _ = cellxgene_census.get_census_version_directory()

    records = collect_proxy_requests()

    # Expecting a CONNECT request followed by a GET request
    assert len(records) == 2
    assert records[0]["method"] == "CONNECT"
    assert records[1]["method"] == "GET"
    assert records[1]["headers"]["host"] == "census.cellxgene.cziscience.com"
    assert "cellxgene-census-python" in records[1]["headers"]["user-agent"]


def test_download_w_proxy_fixture(
    small_dataset_id: str,
    collect_proxy_requests: Callable[[], list[dict]],
    tmp_path: Path,
    test_specific_useragent: str,
):
    # Use of collect_proxy_requests forces test to use a proxy and will check headers of requests made via that proxy
    adata_path = tmp_path / "adata.h5ad"
    cellxgene_census.download_source_h5ad(small_dataset_id, adata_path.as_posix(), census_version="latest")

    records = collect_proxy_requests()
    check_proxy_records(
        records,
        custom_user_agent=test_specific_useragent,
        min_records=3,  # Should request at least a json and the download
    )


def test_query_w_proxy_fixture(collect_proxy_requests: Callable[[], list[dict]]):
    with cellxgene_census.open_soma(census_version="stable") as census:
        _ = cellxgene_census.get_obs(census, "Mus musculus", coords=slice(100, 300))

    records = collect_proxy_requests()
    check_proxy_records(
        records,
        min_records=5,  # some metadata requests, then a lot of request from tiledb
    )


def test_embedding_headers(collect_proxy_requests: Callable[[], list[dict]]):
    import cellxgene_census.experimental

    CENSUS_VERSION = "2023-12-15"

    embeddings_metadata = cellxgene_census.experimental.get_all_available_embeddings(CENSUS_VERSION)
    metadata = embeddings_metadata[0]
    embedding_uri = (
        f"s3://cellxgene-contrib-public/contrib/cell-census/soma/{metadata['census_version']}/{metadata['id']}"
    )
    _ = cellxgene_census.experimental.get_embedding(
        CENSUS_VERSION,
        embedding_uri=embedding_uri,
        obs_soma_joinids=np.arange(100),
    )

    check_proxy_records(collect_proxy_requests())


def test_dataloader_headers(collect_proxy_requests) -> None:
    import cellxgene_census
    from cellxgene_census.experimental.ml.pytorch import ExperimentDataPipe

    soma_experiment = cellxgene_census.open_soma(census_version="latest")["census_data"]["homo_sapiens"]
    dp = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["cell_type"],
        shuffle=False,
    )
    _ = next(iter(dp))

    records = collect_proxy_requests()
    check_proxy_records(records, min_records=5)
