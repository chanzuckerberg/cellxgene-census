from typing import TYPE_CHECKING

import pytest

import cellxgene_census

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="session")
def small_dataset_id() -> str:
    # TODO: REMOVE, copied from test_open
    with cellxgene_census.open_soma(census_version="latest") as census:
        census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()

    small_dataset = census_datasets.nsmallest(1, "dataset_total_cell_count").iloc[0]
    assert isinstance(small_dataset.dataset_id, str)
    return small_dataset.dataset_id


# _BASE_CERT_DIR = "/home/ubuntu/github/proxy.py"

# class TestProxyPyEmbedded(proxy.TestCase):

#     PROXY_PY_STARTUP_FLAGS = [
#         '--num-workers', '1',
#         '--num-acceptors', '1',
#         '--plugin', 'cellxgene_census._testing.ProxyPlugin',
#         '--ca-key-file', f"{_BASE_CERT_DIR}/ca-key.pem",
#         '--ca-cert-file', f"{_BASE_CERT_DIR}/ca-cert.pem",
#         '--ca-signing-key-file', f"{_BASE_CERT_DIR}/ca-signing-key.pem",
#     ]

#     @pytest.fixture(autouse=True)
#     def prepare_fixtures(self, tmp_path: Path, small_dataset_id: str, proxied_soma_context: "soma.options.SOMATileDBContext", http_urls: None):
#         import cellxgene_census._release_directory
#         self.tmp_path = tmp_path
#         self.small_dataset_id = small_dataset_id
#         self.lts_census = cellxgene_census.open_soma(census_version="stable", context=proxied_soma_context)

#     def test_download_w_proxy(self):
#         adata_path = self.tmp_path / "adata.h5ad"
#         cellxgene_census.download_source_h5ad(self.small_dataset_id, adata_path.as_posix(), census_version="latest")
#         assert adata_path.exists() and adata_path.is_file()
#         ad = anndata.read_h5ad(adata_path.as_posix())
#         assert False
#         assert ad is not None

#     def test_query_w_proxy(self):
#         # with pytest.MonkeyPatch.context() as mp:
#             # mp.setenv("HTTPS_PROXY", "http://localhost:8899")
#             # mp.setitem(cellxgene_census._open.DEFAULT_TILEDB_CONFIGURATION, "vfs.s3.verify_ssl", "false")
#             # mp.setitem(cellxgene_census._open.DEFAULT_TILEDB_CONFIGURATION, "vfs.s3.scheme", "http")
#         _ = cellxgene_census.get_obs(self.lts_census, "Mus musculus", coords=slice(100, 300))
#         assert False


# def test_my_application_with_proxy(self) -> None:
#     self.assertTrue(True)


def test_download_w_proxy_fixture(small_dataset_id, proxy_server, tmp_path):
    adata_path = tmp_path / "adata.h5ad"
    cellxgene_census.download_source_h5ad(small_dataset_id, adata_path.as_posix(), census_version="latest")
    assert False


def test_query_w_proxy_fixture(proxy_server):
    with cellxgene_census.open_soma(census_version="stable", context=proxy_server.soma_context) as census:
        _ = cellxgene_census.get_obs(census, "Mus musculus", coords=slice(100, 300))
    assert False
