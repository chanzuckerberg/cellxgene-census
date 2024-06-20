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


def test_download_w_proxy_fixture(small_dataset_id, proxy_instance, tmp_path):
    # Use of proxy_instance forces test to use a proxy and will check headers of requests made via that proxy
    adata_path = tmp_path / "adata.h5ad"
    cellxgene_census.download_source_h5ad(small_dataset_id, adata_path.as_posix(), census_version="latest")


def test_query_w_proxy_fixture(proxy_instance):
    with cellxgene_census.open_soma(census_version="stable") as census:
        _ = cellxgene_census.get_obs(census, "Mus musculus", coords=slice(100, 300))
