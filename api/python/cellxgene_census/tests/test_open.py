import os
import pathlib
import re
import time
from unittest.mock import patch

import anndata
import numpy as np
import pytest
import requests_mock as rm
import tiledb
import tiledbsoma as soma

import cellxgene_census
from cellxgene_census._open import DEFAULT_TILEDB_CONFIGURATION
from cellxgene_census._release_directory import CELL_CENSUS_RELEASE_DIRECTORY_URL


@pytest.mark.live_corpus
def test_open_soma_stable() -> None:
    # There should _always_ be a 'stable'
    with cellxgene_census.open_soma(census_version="stable") as census:
        assert census is not None
        assert isinstance(census, soma.Collection)

    # and it should be the latest, until the first "stable" build is available
    with cellxgene_census.open_soma() as default_census:
        assert default_census.uri == census.uri
        for k, v in DEFAULT_TILEDB_CONFIGURATION.items():
            assert census.context.tiledb_ctx.config()[k] == str(v)

    # TODO: After the first "stable" build is available, this commented-out code can be replace this above block
    # and it should always be the default
    # with cellxgene_census.open_soma() as default_census:
    #     assert default_census.uri == census.uri
    #     for k, v in DEFAULT_TILEDB_CONFIGURATION.items():
    #         assert census.context.tiledb_ctx.config()[k] == str(v)


@pytest.mark.live_corpus
def test_open_soma_latest() -> None:
    # There should _always_ be a 'latest'
    with cellxgene_census.open_soma(census_version="latest") as census:
        assert census is not None
        assert isinstance(census, soma.Collection)


@pytest.mark.live_corpus
def test_open_soma_with_context() -> None:
    description = cellxgene_census.get_census_version_description("latest")
    uri = description["soma"]["uri"]
    s3_region = description["soma"].get("s3_region")
    assert s3_region == "us-west-2"

    # Verify the default region is set correctly in the TileDB context object.
    with cellxgene_census.open_soma(census_version="latest", context=soma.SOMATileDBContext()) as census:
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
    with cellxgene_census.open_soma(uri=uri, context=context) as census:
        assert census.uri == uri
        assert census.context.tiledb_ctx.config()["soma.init_buffer_bytes"] == soma_init_buffer_bytes
        assert census.context.timestamp_ms == timestamp_ms


def test_open_soma_invalid_args() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Must specify either a census version or an explicit URI."),
    ):
        cellxgene_census.open_soma(census_version=None)


def test_open_soma_errors(requests_mock: rm.Mocker) -> None:
    requests_mock.get(CELL_CENSUS_RELEASE_DIRECTORY_URL, json={})
    with pytest.raises(
        ValueError,
        match=re.escape(
            'The "does-not-exist" Census version is not valid. Use get_census_version_directory() to retrieve available versions.'
        ),
    ):
        cellxgene_census.open_soma(census_version="does-not-exist")


def test_open_soma_defaults_to_latest_if_missing_stable(requests_mock: rm.Mocker) -> None:
    dir_missing_stable = {
        "latest": "2022-11-01",
        "2022-11-01": {
            "release_date": "2022-11-30",
            "release_build": "2022-11-01",
            "soma": {
                "uri": "s3://cellxgene-data-public/cell-census/2022-11-01/soma/",
                "s3_region": "us-west-2",
            },
            "h5ads": {
                "uri": "s3://cellxgene-data-public/cell-census/2022-11-01/h5ads/",
                "s3_region": "us-west-2",
            },
        },
    }

    requests_mock.get(CELL_CENSUS_RELEASE_DIRECTORY_URL, json=dir_missing_stable)
    with patch("cellxgene_census._open._open_soma") as m:
        cellxgene_census.open_soma(census_version="stable")
        m.assert_called_once_with(
            {"uri": "s3://cellxgene-data-public/cell-census/2022-11-01/soma/", "s3_region": "us-west-2"}, None
        )


def test_open_soma_defaults_to_stable(requests_mock: rm.Mocker) -> None:
    directory_with_stable = {
        "stable": "2022-10-01",
        "2022-10-01": {
            "release_date": "2022-10-30",
            "release_build": "2022-10-01",
            "soma": {
                "uri": "s3://cellxgene-data-public/cell-census/2022-10-01/soma/",
                "s3_region": "us-west-2",
            },
            "h5ads": {
                "uri": "s3://cellxgene-data-public/cell-census/2022-10-01/h5ads/",
                "s3_region": "us-west-2",
            },
        },
    }

    requests_mock.get(CELL_CENSUS_RELEASE_DIRECTORY_URL, json=directory_with_stable)
    with patch("cellxgene_census._open._open_soma") as m:
        cellxgene_census.open_soma()
        m.assert_called_once_with(
            {"uri": "s3://cellxgene-data-public/cell-census/2022-10-01/soma/", "s3_region": "us-west-2"}, None
        )


@pytest.mark.live_corpus
def test_get_source_h5ad_uri() -> None:
    with cellxgene_census.open_soma(census_version="latest") as census:
        census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()

    rng = np.random.default_rng()
    for idx in rng.choice(np.arange(len(census_datasets)), size=3, replace=False):
        a_dataset = census_datasets.iloc[idx]
        locator = cellxgene_census.get_source_h5ad_uri(a_dataset.dataset_id, census_version="latest")
        assert isinstance(locator, dict)
        assert "uri" in locator
        assert locator["uri"].endswith(a_dataset.dataset_h5ad_path)


def test_get_source_h5ad_uri_errors() -> None:
    with pytest.raises(KeyError):
        cellxgene_census.get_source_h5ad_uri(dataset_id="no/such/id")


@pytest.fixture
def small_dataset_id() -> str:
    with cellxgene_census.open_soma(census_version="latest") as census:
        census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()

    small_dataset = census_datasets.nsmallest(1, "dataset_total_cell_count").iloc[0]
    assert isinstance(small_dataset.dataset_id, str)
    return small_dataset.dataset_id


@pytest.mark.live_corpus
def test_download_source_h5ad(tmp_path: pathlib.Path, small_dataset_id: str) -> None:
    adata_path = tmp_path / "adata.h5ad"
    cellxgene_census.download_source_h5ad(small_dataset_id, adata_path.as_posix(), census_version="latest")
    assert adata_path.exists() and adata_path.is_file()
    ad = anndata.read_h5ad(adata_path.as_posix())
    assert ad is not None


def test_download_source_h5ad_errors(tmp_path: pathlib.Path, small_dataset_id: str) -> None:
    existing_file = tmp_path / "existing_file.h5ad"
    existing_file.touch()
    assert existing_file.exists()

    with pytest.raises(ValueError):
        cellxgene_census.download_source_h5ad(small_dataset_id, existing_file.as_posix(), census_version="latest")

    with pytest.raises(ValueError):
        cellxgene_census.download_source_h5ad(small_dataset_id, "/tmp/dirname/", census_version="latest")


@pytest.mark.live_corpus
def test_opening_census_without_anon_access_fails_with_bogus_creds() -> None:
    os.environ["AWS_ACCESS_KEY_ID"] = "fake_id"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "fake_key"
    # Passing an empty context
    with pytest.raises(tiledb.TileDBError, match=r"The AWS Access Key Id you provided does not exist in our records"):
        cellxgene_census.open_soma(census_version="latest", context=soma.SOMATileDBContext())


@pytest.mark.live_corpus
def test_can_open_with_anonymous_access() -> None:
    """
    With anonymous access, `open_soma` must be able to access the census even with bogus credentials
    """
    os.environ["AWS_ACCESS_KEY_ID"] = "fake_id"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "fake_key"
    with cellxgene_census.open_soma(census_version="latest") as census:
        assert census is not None
        assert isinstance(census, soma.Collection)
