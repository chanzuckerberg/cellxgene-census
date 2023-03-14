import io
import pathlib
import re
from unittest.mock import patch

import pytest

from tools.cell_census_builder.manifest import CXG_BASE_URI, load_manifest


@pytest.fixture
def manifest_csv(tmp_path: pathlib.Path) -> io.TextIOWrapper:
    # assets_path = f"{tmp_path}/h5ads"
    # os.mkdir(assets_path)
    manifest_content = """
    dataset_id_1, data/h5ads/dataset_id_1.h5ad
    dataset_id_2, data/h5ads/dataset_id_2.h5ad
    """
    path = f"{tmp_path}/manifest.csv"
    h5ad_path = f"{tmp_path}/data/h5ads/"
    pathlib.Path(h5ad_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path("data/h5ads/dataset_id_1.h5ad").touch()
    pathlib.Path("data/h5ads/dataset_id_2.h5ad").touch()
    with open(path, "w+") as f:
        f.writelines(manifest_content.strip())

    return open(path)


def test_load_manifest_from_file(manifest_csv: io.TextIOWrapper) -> None:
    """
    If specified a parameter, `load_manifest` should load the dataset manifest from such file.
    """
    manifest = load_manifest(manifest_csv)
    assert len(manifest) == 2
    assert manifest[0].dataset_id == "dataset_id_1"
    assert manifest[1].dataset_id == "dataset_id_2"
    assert manifest[0].corpora_asset_h5ad_uri == "data/h5ads/dataset_id_1.h5ad"
    assert manifest[1].corpora_asset_h5ad_uri == "data/h5ads/dataset_id_2.h5ad"


def test_load_manifest_from_cxg() -> None:
    """
    If no parameters are specified, `load_manifest` should load the dataset list from Discover API.
    """
    with patch("tools.cell_census_builder.manifest.fetch_json") as m:

        def mock_call_fn(uri):  # type: ignore
            if uri == f"{CXG_BASE_URI}curation/v1/collections":
                return [
                    {
                        "id": "collection_1",
                        "doi": None,
                        "name": "1",
                        "datasets": [{"id": "dataset_id_1"}, {"id": "dataset_id_2"}],
                    }
                ]
            elif m := re.match(rf"{CXG_BASE_URI}curation/v1/collections/(\w+)/datasets/(\w+)$", uri):
                return {"id": m[2], "schema_version": "3.0.0", "title": f"dataset #{m[2]}"}
            elif m := re.match(rf"{CXG_BASE_URI}curation/v1/collections/(\w+)/datasets/(\w+)/assets$", uri):
                return [{"filetype": "H5AD", "filesize": 1024, "presigned_url": f"https://fake.url/{m[2]}.h5ad"}]

        m.side_effect = mock_call_fn

        manifest = load_manifest(None)
        assert len(manifest) == 2
        assert manifest[0].dataset_id == "dataset_id_1"
        assert manifest[1].dataset_id == "dataset_id_2"
        assert manifest[0].corpora_asset_h5ad_uri == "https://fake.url/dataset_id_1.h5ad"
        assert manifest[1].corpora_asset_h5ad_uri == "https://fake.url/dataset_id_2.h5ad"


def test_load_manifest_from_cxg_excludes_datasets_with_old_schema() -> None:
    """
    `load_manifest` should exclude datasets that do not have a current schema version.
    """
    with patch("tools.cell_census_builder.manifest.fetch_json") as m:

        def mock_call_fn(uri):  # type: ignore
            if uri == f"{CXG_BASE_URI}curation/v1/collections":
                return [
                    {
                        "id": "collection_1",
                        "doi": None,
                        "name": "1",
                        "datasets": [{"id": "dataset_id_1"}, {"id": "dataset_id_2"}],
                    }
                ]
            elif m := re.match(rf"{CXG_BASE_URI}curation/v1/collections/(\w+)/datasets/(\w+)$", uri):
                return {
                    "id": m[2],
                    "schema_version": "3.0.0" if m[2] == "dataset_id_1" else "2.0.0",
                    "title": f"dataset #{m[2]}",
                }
            elif m := re.match(rf"{CXG_BASE_URI}curation/v1/collections/(\w+)/datasets/(\w+)/assets$", uri):
                return [{"filetype": "H5AD", "filesize": 1024, "presigned_url": f"https://fake.url/{m[2]}.h5ad"}]

        m.side_effect = mock_call_fn

        manifest = load_manifest(None)
        assert len(manifest) == 1
        assert manifest[0].dataset_id == "dataset_id_1"
        assert manifest[0].corpora_asset_h5ad_uri == "https://fake.url/dataset_id_1.h5ad"


def test_load_manifest_from_cxg_excludes_datasets_with_no_assets() -> None:
    """
    `load_manifest` should exclude datasets that do not have assets
    """
    with patch("tools.cell_census_builder.manifest.fetch_json") as m:

        def mock_call_fn(uri):  # type: ignore
            if uri == f"{CXG_BASE_URI}curation/v1/collections":
                return [
                    {
                        "id": "collection_1",
                        "doi": None,
                        "name": "1",
                        "datasets": [{"id": "dataset_id_1"}, {"id": "dataset_id_2"}],
                    }
                ]
            elif m := re.match(rf"{CXG_BASE_URI}curation/v1/collections/(\w+)/datasets/(\w+)$", uri):
                return {"id": m[2], "schema_version": "3.0.0", "title": f"dataset #{m[2]}"}
            elif m := re.match(rf"{CXG_BASE_URI}curation/v1/collections/(\w+)/datasets/dataset_id_1/assets$", uri):
                return [{"filetype": "H5AD", "filesize": 1024, "presigned_url": "https://fake.url/dataset_id_1.h5ad"}]
            elif m := re.match(rf"{CXG_BASE_URI}curation/v1/collections/(\w+)/datasets/dataset_id_2/assets$", uri):
                return []

        m.side_effect = mock_call_fn

        manifest = load_manifest(None)
        assert len(manifest) == 1
        assert manifest[0].dataset_id == "dataset_id_1"
        assert manifest[0].corpora_asset_h5ad_uri == "https://fake.url/dataset_id_1.h5ad"
