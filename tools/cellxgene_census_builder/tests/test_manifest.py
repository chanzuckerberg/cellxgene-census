import logging
import pathlib
import uuid
from unittest.mock import patch

import fsspec
import pytest
from cellxgene_census_builder.build_soma.manifest import load_manifest
from cellxgene_census_builder.build_state import CensusBuildConfig


def test_load_manifest_from_file(tmp_path: pathlib.Path, manifest_csv: str, empty_blocklist: str) -> None:
    """
    If specified a parameter, `load_manifest` should load the dataset manifest from such file.
    """
    manifest = load_manifest(manifest_csv, empty_blocklist)
    assert len(manifest) == 2
    assert manifest[0].dataset_id == "dataset_id_1"
    assert manifest[1].dataset_id == "dataset_id_2"
    assert manifest[0].dataset_asset_h5ad_uri == f"{tmp_path}/data/h5ads/dataset_id_1.h5ad"
    assert manifest[1].dataset_asset_h5ad_uri == f"{tmp_path}/data/h5ads/dataset_id_2.h5ad"

    with open(manifest_csv) as fp:
        manifest = load_manifest(fp, empty_blocklist)
        assert len(manifest) == 2
        assert manifest[0].dataset_id == "dataset_id_1"
        assert manifest[1].dataset_id == "dataset_id_2"
        assert manifest[0].dataset_asset_h5ad_uri == f"{tmp_path}/data/h5ads/dataset_id_1.h5ad"
        assert manifest[1].dataset_asset_h5ad_uri == f"{tmp_path}/data/h5ads/dataset_id_2.h5ad"


def test_load_manifest_does_dedup(manifest_csv_with_duplicates: str, empty_blocklist: str) -> None:
    """
    `load_manifest` should not include duplicate datasets from the manifest
    """
    manifest = load_manifest(manifest_csv_with_duplicates, empty_blocklist)
    assert len(manifest) == 2

    with open(manifest_csv_with_duplicates) as fp:
        manifest = load_manifest(fp, empty_blocklist)
        assert len(manifest) == 2


def test_manifest_dataset_block(tmp_path: pathlib.Path, manifest_csv: str, empty_blocklist: str) -> None:
    # grab first item from the manifest, and block it.
    with open(manifest_csv) as f:
        first_dataset_id = f.readline().split(",")[0].strip()

    blocklist_fname = f"{tmp_path}/blocklist.txt"
    blocklist_content = f"# a comment\n\n{first_dataset_id}\n\n"
    with open(blocklist_fname, "w+") as f:
        f.writelines(blocklist_content.strip())

    manifest = load_manifest(manifest_csv, blocklist_fname)
    assert len(manifest) == 1
    assert not any(d.dataset_id == first_dataset_id for d in manifest)


def test_load_manifest_from_cxg(empty_blocklist: str) -> None:
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
                "citation": "citation",
                "title": "dataset #1",
                "schema_version": "4.0.0",
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
                "dataset_version_id": "dataset_id_1",
                "cell_count": 10,
                "mean_genes_per_cell": 99.9,
            },
            {
                "dataset_id": "dataset_id_2",
                "collection_id": "collection_1",
                "collection_name": "1",
                "collection_doi": None,
                "citation": "citation",
                "title": "dataset #2",
                "schema_version": "4.0.0",
                "assets": [{"filesize": 456, "filetype": "H5AD", "url": "https://fake.url/dataset_id_2.h5ad"}],
                "dataset_version_id": "dataset_id_2",
                "cell_count": 11,
                "mean_genes_per_cell": 109.1,
            },
        ]

        manifest = load_manifest(None, empty_blocklist)
        assert len(manifest) == 2
        assert manifest[0].dataset_id == "dataset_id_1"
        assert manifest[1].dataset_id == "dataset_id_2"
        assert manifest[0].dataset_asset_h5ad_uri == "https://fake.url/dataset_id_1.h5ad"
        assert manifest[0].asset_h5ad_filesize == 123
        assert manifest[1].dataset_asset_h5ad_uri == "https://fake.url/dataset_id_2.h5ad"
        assert manifest[1].asset_h5ad_filesize == 456


def test_load_manifest_from_cxg_errors_on_datasets_with_old_schema(
    caplog: pytest.LogCaptureFixture, empty_blocklist: str
) -> None:
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
                "citation": "citation",
                "title": "dataset #1",
                "schema_version": "4.0.0",
                "assets": [{"filesize": 123, "filetype": "H5AD", "url": "https://fake.url/dataset_id_1.h5ad"}],
                "dataset_version_id": "dataset_id_1",
                "cell_count": 10,
                "mean_genes_per_cell": 99.9,
            },
            {
                "dataset_id": "dataset_id_2",
                "collection_id": "collection_1",
                "collection_name": "1",
                "collection_doi": None,
                "citation": "citation",
                "title": "dataset #2",
                "schema_version": "2.0.0",  # Old schema version
                "assets": [{"filesize": 456, "filetype": "H5AD", "url": "https://fake.url/dataset_id_2.h5ad"}],
                "dataset_version_id": "dataset_id_2",
                "cell_count": 10,
                "mean_genes_per_cell": 99.9,
            },
        ]

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match=r"unsupported schema version"):
                load_manifest(None, empty_blocklist)

            for record in caplog.records:
                assert record.levelname == "ERROR"
                assert "unsupported schema version" in record.msg


def test_load_manifest_from_cxg_excludes_datasets_with_no_assets(
    caplog: pytest.LogCaptureFixture, empty_blocklist: str
) -> None:
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
                "citation": "citation",
                "title": "dataset #1",
                "schema_version": "4.0.0",
                "assets": [{"filesize": 123, "filetype": "H5AD", "url": "https://fake.url/dataset_id_1.h5ad"}],
                "dataset_version_id": "dataset_id_1",
                "cell_count": 10,
                "mean_genes_per_cell": 99.9,
            },
            {
                "dataset_id": "dataset_id_2",
                "collection_id": "collection_1",
                "collection_name": "1",
                "collection_doi": None,
                "citation": "citation",
                "title": "dataset #2",
                "schema_version": "4.0.0",
                "assets": [],
                "dataset_version_id": "dataset_id_2",
                "cell_count": 10,
                "mean_genes_per_cell": 99.9,
            },
        ]

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match=r"unable to find H5AD asset"):
                load_manifest(None, empty_blocklist)

            for record in caplog.records:
                assert record.levelname == "ERROR"
                assert "unable to find H5AD asset" in record.msg


def test_blocklist_alive_and_well() -> None:
    """
    Perform three checks:
    1. Block list is specified in the default configuration
    2. The file exists at the specified location
    3. The file "looks like" a block list
    """

    config = CensusBuildConfig()

    assert config.dataset_id_blocklist_uri

    dataset_id_blocklist_uri = config.dataset_id_blocklist_uri

    # test for existance by reading it. NOTE: if the file moves, this test will fail until
    # the new file location is merged to main.
    with fsspec.open(dataset_id_blocklist_uri, "rt") as fp:
        for line in fp:
            # each line must be a comment, blank or a UUID
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # UUID() raises ValueError upon malformed UUID
            # Equality check enforces formatting (i.e., dashes)
            assert line == str(uuid.UUID(hex=line))
