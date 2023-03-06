import pathlib
import time

import anndata
import numpy as np
import pytest
import tiledbsoma as soma

import cell_census
from cell_census._open import DEFAULT_TILEDB_CONFIGURATION


@pytest.mark.live_corpus
def test_open_soma_latest() -> None:
    # There should _always_ be a 'latest'
    with cell_census.open_soma(census_version="latest") as census:
        assert census is not None
        assert isinstance(census, soma.Collection)

    # and it should always be the default
    with cell_census.open_soma() as default_census:
        assert default_census.uri == census.uri
        for k, v in DEFAULT_TILEDB_CONFIGURATION.items():
            assert census.context.tiledb_ctx.config()[k] == str(v)


@pytest.mark.live_corpus
def test_open_soma_with_context() -> None:
    description = cell_census.get_census_version_description("latest")
    uri = description["soma"]["uri"]
    s3_region = description["soma"].get("s3_region")
    assert s3_region == "us-west-2"

    # Verify the default region is set correctly in the TileDB context object.
    with cell_census.open_soma(census_version="latest", context=soma.SOMATileDBContext()) as census:
        assert census.context.tiledb_ctx.config()["vfs.s3.region"] == s3_region

    # Verify that config provided is passed through correctly
    soma_init_buffer_bytes = "221000"
    timestamp_ms = int(time.time() * 1000) - 10  # don't use exactly current time, as that is the default
    cfg = {
        "timestamp": timestamp_ms,
        "tiledb_config": {
            "soma.init_buffer_bytes": soma_init_buffer_bytes,
            "vfs.s3.region": s3_region,
        },
    }
    context = soma.SOMATileDBContext().replace(**cfg)
    with cell_census.open_soma(uri=uri, context=context) as census:
        assert census.uri == uri
        assert census.context.tiledb_ctx.config()["soma.init_buffer_bytes"] == soma_init_buffer_bytes
        assert census.context.timestamp_ms == timestamp_ms


def test_open_soma_errors() -> None:
    with pytest.raises(ValueError):
        cell_census.open_soma(census_version=None)


@pytest.mark.live_corpus
def test_get_source_h5ad_uri() -> None:
    with cell_census.open_soma(census_version="latest") as census:
        census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()

    rng = np.random.default_rng()
    for idx in rng.choice(np.arange(len(census_datasets)), size=10, replace=False):
        a_dataset = census_datasets.iloc[idx]
        locator = cell_census.get_source_h5ad_uri(a_dataset.dataset_id)
        assert isinstance(locator, dict)
        assert "uri" in locator
        assert locator["uri"].endswith(a_dataset.dataset_h5ad_path)


def test_get_source_h5ad_uri_errors() -> None:
    with pytest.raises(KeyError):
        cell_census.get_source_h5ad_uri(dataset_id="no/such/id")


@pytest.fixture
def small_dataset_id() -> str:
    with cell_census.open_soma(census_version="latest") as census:
        census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()

    small_dataset = census_datasets.nsmallest(1, "dataset_total_cell_count").iloc[0]
    assert isinstance(small_dataset.dataset_id, str)
    return small_dataset.dataset_id


@pytest.mark.live_corpus
def test_download_source_h5ad(tmp_path: pathlib.Path, small_dataset_id: str) -> None:
    adata_path = tmp_path / "adata.h5ad"
    cell_census.download_source_h5ad(small_dataset_id, adata_path.as_posix(), census_version="latest")
    assert adata_path.exists() and adata_path.is_file()
    ad = anndata.read_h5ad(adata_path.as_posix())
    assert ad is not None


def test_download_source_h5ad_errors(tmp_path: pathlib.Path, small_dataset_id: str) -> None:
    existing_file = tmp_path / "existing_file.h5ad"
    existing_file.touch()
    assert existing_file.exists()

    with pytest.raises(ValueError):
        cell_census.download_source_h5ad(small_dataset_id, existing_file.as_posix(), census_version="latest")

    with pytest.raises(ValueError):
        cell_census.download_source_h5ad(small_dataset_id, "/tmp/dirname/", census_version="latest")
