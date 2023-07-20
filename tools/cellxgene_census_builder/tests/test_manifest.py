import pathlib
from unittest.mock import patch

import pytest
from cellxgene_census_builder.build_soma.manifest import load_manifest


def test_load_manifest_from_file(tmp_path: pathlib.Path, manifest_csv: str) -> None:
    """
    If specified a parameter, `load_manifest` should load the dataset manifest from such file.
    """
    manifest = load_manifest(manifest_csv)
    assert len(manifest) == 2
    assert manifest[0].dataset_id == "dataset_id_1"
    assert manifest[1].dataset_id == "dataset_id_2"
    assert manifest[0].dataset_asset_h5ad_uri == f"{tmp_path}/data/h5ads/dataset_id_1.h5ad"
    assert manifest[1].dataset_asset_h5ad_uri == f"{tmp_path}/data/h5ads/dataset_id_2.h5ad"

    with open(manifest_csv) as fp:
        manifest = load_manifest(fp)
        assert len(manifest) == 2
        assert manifest[0].dataset_id == "dataset_id_1"
        assert manifest[1].dataset_id == "dataset_id_2"
        assert manifest[0].dataset_asset_h5ad_uri == f"{tmp_path}/data/h5ads/dataset_id_1.h5ad"
        assert manifest[1].dataset_asset_h5ad_uri == f"{tmp_path}/data/h5ads/dataset_id_2.h5ad"


def test_load_manifest_does_dedup(manifest_csv_with_duplicates: str) -> None:
    """
    `load_manifest` should not include duplicate datasets from the manifest
    """
    manifest = load_manifest(manifest_csv_with_duplicates)
    assert len(manifest) == 2

    with open(manifest_csv_with_duplicates) as fp:
        manifest = load_manifest(fp)
        assert len(manifest) == 2


def test_load_manifest_from_cxg() -> None:
    """
    If no parameters are specified, `load_manifest` should load the dataset list from Discover API.
    """
    with patch("cellxgene_census_builder.build_soma.manifest.fetch_json") as m:
        m.return_value = [
            {
                "dataset_id": "dataset_id_1",
                "collection_id": "collection_1",
                "collection_name": "1",
                "collection_doi": None,
                "title": "dataset #1",
                "schema_version": "3.0.0",
                "assets": [
                    {
                        "filesize": 123,
                        "filetype": "H5AD",
                        "url": "https://fake.url/dataset_id_1.h5ad",
                    },
                    {
                        "filesize": 234,
                        "filetype": "RDS",
                        "url": "https://fake.url/dataset_id_1.rds",
                    },
                ],
            },
            {
                "dataset_id": "dataset_id_2",
                "collection_id": "collection_1",
                "collection_name": "1",
                "collection_doi": None,
                "title": "dataset #2",
                "schema_version": "3.0.0",
                "assets": [
                    {
                        "filesize": 456,
                        "filetype": "H5AD",
                        "url": "https://fake.url/dataset_id_2.h5ad",
                    }
                ],
            },
        ]

        manifest = load_manifest(None)
        assert len(manifest) == 2
        assert manifest[0].dataset_id == "dataset_id_1"
        assert manifest[1].dataset_id == "dataset_id_2"
        assert manifest[0].dataset_asset_h5ad_uri == "https://fake.url/dataset_id_1.h5ad"
        assert manifest[0].asset_h5ad_filesize == 123
        assert manifest[1].dataset_asset_h5ad_uri == "https://fake.url/dataset_id_2.h5ad"
        assert manifest[1].asset_h5ad_filesize == 456


def test_load_manifest_from_cxg_errors_on_datasets_with_old_schema() -> None:
    """
    `load_manifest` should exclude datasets that do not have a current schema version.
    """
    with patch("cellxgene_census_builder.build_soma.manifest.fetch_json") as m:
        m.return_value = [
            {
                "dataset_id": "dataset_id_1",
                "collection_id": "collection_1",
                "collection_name": "1",
                "collection_doi": None,
                "title": "dataset #1",
                "schema_version": "3.0.0",
                "assets": [
                    {
                        "filesize": 123,
                        "filetype": "H5AD",
                        "url": "https://fake.url/dataset_id_1.h5ad",
                    }
                ],
            },
            {
                "dataset_id": "dataset_id_2",
                "collection_id": "collection_1",
                "collection_name": "1",
                "collection_doi": None,
                "title": "dataset #2",
                "schema_version": "2.0.0",  # Old schema version
                "assets": [
                    {
                        "filesize": 456,
                        "filetype": "H5AD",
                        "url": "https://fake.url/dataset_id_2.h5ad",
                    }
                ],
            },
        ]

        with pytest.raises(RuntimeError, match=r"unsupported schema version"):
            load_manifest(None)


def test_load_manifest_from_cxg_excludes_datasets_with_no_assets() -> None:
    """
    `load_manifest` should raise error if it finds datasets without assets
    """
    with patch("cellxgene_census_builder.build_soma.manifest.fetch_json") as m:
        m.return_value = [
            {
                "dataset_id": "dataset_id_1",
                "collection_id": "collection_1",
                "collection_name": "1",
                "collection_doi": None,
                "title": "dataset #1",
                "schema_version": "3.0.0",
                "assets": [
                    {
                        "filesize": 123,
                        "filetype": "H5AD",
                        "url": "https://fake.url/dataset_id_1.h5ad",
                    }
                ],
            },
            {
                "dataset_id": "dataset_id_2",
                "collection_id": "collection_1",
                "collection_name": "1",
                "collection_doi": None,
                "title": "dataset #2",
                "schema_version": "3.0.0",
                "assets": [],
            },
        ]

        with pytest.raises(RuntimeError):
            load_manifest(None)
