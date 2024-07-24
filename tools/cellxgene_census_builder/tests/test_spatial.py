import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cellxgene_census
import pandas as pd
import pooch
import pytest
import tiledbsoma
from filelock import FileLock

from cellxgene_census_builder.build_soma.manifest import load_manifest


@dataclass
class SpatialBuild:
    soma_path: Path
    manifest_path: Path
    blocklist: Path


# These should be updated with every schema update
VISIUM_DATASET_URIS = [
    "https://datasets.cellxgene.cziscience.com/6811c454-def2-4d9e-b360-aa8a69f843ce.h5ad",
    "https://datasets.cellxgene.cziscience.com/17d9e43f-1251-4f1e-8a5b-a96f2c89e5ec.h5ad",
]


def create_manifest_csv_file(spatial_datasets_dir: Path | str, manifest_file_path: Path | str) -> None:
    file_ids = [os.path.splitext(filename)[0] for filename in os.listdir(spatial_datasets_dir)]
    file_paths = [os.path.join(spatial_datasets_dir, filename) for filename in os.listdir(spatial_datasets_dir)]
    manifest_content = "\n".join([", ".join(pair) for pair in zip(file_ids, file_paths, strict=False)])

    with open(manifest_file_path, "w") as f:
        f.write(manifest_content.strip())


def _fetch_datasets(h5ad_dir: Path, manifest_pth: str) -> None:
    for uri in VISIUM_DATASET_URIS:
        # output_pth = uri.split("/")[-1]
        pooch.retrieve(uri, None, path=h5ad_dir)
    create_manifest_csv_file(h5ad_dir, manifest_pth)


@pytest.fixture(scope="session")
def spatial_manifest(tmp_path_factory, worker_id) -> Path:
    # if worker_id == "master":
    #     # not executing in with multiple workers, just produce the data and let
    #     # pytest's fixture caching do its job
    #     return _fetch_datasets(anndata_dir, manifest_pth)

    # get the temp directory shared by all workers
    # root_tmp_dir = tmp_path_factory.getbasetemp().parent
    root_tmp_dir = tmp_path_factory.getbasetemp()  # Not shared, but also not reused
    anndata_dir = pooch.os_cache("cellxgene_census_builder")
    manifest_pth = root_tmp_dir / "manifest.csv"
    # fn = root_tmp_dir / "data.json"
    # TODO: use pooch for downloading files so I can cache them
    with FileLock(str(manifest_pth) + ".lock"):
        if manifest_pth.is_file():
            pass
        else:
            anndata_dir.mkdir(exist_ok=True)
            _fetch_datasets(anndata_dir, manifest_pth)
    return manifest_pth


@pytest.fixture(scope="session")
def spatial_build(spatial_manifest, tmp_path_factory) -> SpatialBuild:
    root_tmp_dir = tmp_path_factory.getbasetemp()
    census_dir = root_tmp_dir / "census-builds"
    build_tag = "test-spatial-build"
    blocklist = root_tmp_dir / "block.csv"
    with blocklist.open("w") as _:
        pass
    # Returns URI for build object
    subprocess.run(
        [
            "python",
            "-m",
            "cellxgene_census_builder.build_soma",
            "-v",
            "--build-tag",
            build_tag,
            str(census_dir),
            "build",
            "--manifest",
            str(spatial_manifest),
        ],
        check=True,
    )

    return SpatialBuild(
        census_dir / build_tag / "soma",
        spatial_manifest,
        blocklist,
    )


def test_spatial_build(spatial_build):
    manifest = load_manifest(str(spatial_build.manifest_path), str(spatial_build.blocklist))
    census = cellxgene_census.open_soma(uri=str(spatial_build.soma_path))
    obs = census["census_spatial"]["homo_sapiens"].obs.read().concat().to_pandas()
    assert set(obs["dataset_id"].unique()) == {d.dataset_id for d in manifest}


def _to_df(somadf: tiledbsoma.DataFrame) -> pd.DataFrame:
    return somadf.read().concat().to_pandas()


def test_locations(spatial_build):
    census = cellxgene_census.open_soma(uri=str(spatial_build.soma_path))
    spatial = census["census_spatial"]["homo_sapiens"]
    obs_scene = _to_df(spatial["obs_scene"])

    # Test that the obs_scene join id + dataset work with locations
    for scene_id, subdf in obs_scene.groupby("scene_id"):
        locations = _to_df(spatial["spatial"][scene_id]["obsl"]["loc"])
        assert locations["soma_joinid"].isin(subdf["soma_joinid"]).all()

    # TODO: Test that locations match the anndata
