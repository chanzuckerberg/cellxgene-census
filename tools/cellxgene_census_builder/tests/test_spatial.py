# mypy: ignore-errors
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cellxgene_census
import h5py
import numpy as np
import pandas as pd
import pooch
import pytest
import tiledbsoma
from anndata.io import read_elem
from filelock import FileLock

from cellxgene_census_builder.build_soma.manifest import load_manifest


@dataclass
class SpatialBuild:
    soma_path: Path
    manifest_path: Path
    blocklist: Path


# These should be updated with every schema update
# Fields besides dataset_id are only needed for validation of the built object, and can be dummy values
SPATIAL_TEST_DATASETS = [
    {
        "dataset_id": "fee901ce-87ea-46cd-835a-c15906a4aa6d",
        "collection_id": "21bbfaec-6958-46bc-b1cd-1535752f6304",
        "collection_name": "Single cell and spatial transcriptomics in Sjögren’s Disease-affected Human Salivary Glands",
        "dataset_title": "spRNA-seq for GZMK+CD8+ T cells Target A Specific Acinar Cell Type in Sjögren’s Disease - Block #7",
        "dataset_version_id": "fee901ce-87ea-46cd-835a-c15906a4aa6d",
    },  # Homo sapiens Visium
    {
        "dataset_id": "e944a0f7-e398-4e8f-a060-94dae8a08fb3",
        "collection_id": "68cba939-4e72-4405-80ef-512a05044fba",
        "collection_name": "Profiling the heterogeneity of colorectal cancer consensus molecular subtypes using spatial transcriptomics",
        "dataset_title": "S5_Rec A121573 Rep1",
        "dataset_version_id": "e944a0f7-e398-4e8f-a060-94dae8a08fb3",
    },  # Homo sapiens Visium
    {
        "dataset_id": "db4b5e64-71bd-4ed8-8ec9-21471194485b",
        "collection_id": "a96133de-e951-4e2d-ace6-59db8b3bfb1d",
        "collection_name": "HTAN/HTAPP Broad - Spatio-molecular dissection of the breast cancer metastatic microenvironment",
        "dataset_title": "HTAPP-982-SMP-7629 Slide-seq",
        "dataset_version_id": "db4b5e64-71bd-4ed8-8ec9-21471194485b",
    },  # Homo sapiens Slide-seq
    {
        "dataset_id": "563baeba-7936-4600-b61c-c003f89c8bdb",
        "collection_id": "8e880741-bf9a-4c8e-9227-934204631d2a",
        "collection_name": "High Resolution Slide-seqV2 Spatial Transcriptomics Enables Discovery of Disease-Specific Cell Neighborhoods and Pathways",
        "dataset_title": "Spatial transcriptomics in mouse: Puck_200127_09",
        "dataset_version_id": "563baeba-7936-4600-b61c-c003f89c8bdb",
    },  # Mus musculus Slide-seq w/ repeated coordinates
]


def _fetch_datasets_and_write_manifest(h5ad_dir: Path, manifest_pth: str) -> None:
    for dataset in SPATIAL_TEST_DATASETS:
        dataset_id = dataset["dataset_id"]
        pooch.retrieve(
            f"https://datasets.cellxgene.cziscience.com/{dataset_id}.h5ad",
            None,
            fname=f"{dataset_id}.h5ad",
            path=h5ad_dir,
        )
    table = pd.DataFrame(SPATIAL_TEST_DATASETS)
    table["dataset_asset_h5ad_uri"] = table["dataset_id"].apply(lambda x: str(h5ad_dir / f"{x}.h5ad"))
    table.to_csv(manifest_pth, index=False)


@pytest.fixture(scope="session")
def spatial_manifest(tmp_path_factory, worker_id) -> Path:
    # TODO: We should refactor this fixture to better support running tests in parallel.
    # Currently datasets are downloaded in a thread safe way to prevent overwriting, and build are written to seperate directories
    # However builds can happen in parallel which ends up oversubscribbing the CPU and taking a long time.
    # Ideally we would do a build once and re-use it across workers (as each build takes some time). See these docs for more info on how to do this:
    # https://pytest-xdist.readthedocs.io/en/stable/how-to.html#making-session-scoped-fixtures-execute-only-once
    root_tmp_dir = tmp_path_factory.getbasetemp()  # Not shared, but also not reused
    anndata_dir = pooch.os_cache("cellxgene_census_builder")
    manifest_pth = root_tmp_dir / "manifest.csv"
    with FileLock(str(manifest_pth) + ".lock"):
        if manifest_pth.is_file():
            pass
        else:
            anndata_dir.mkdir(exist_ok=True)
            _fetch_datasets_and_write_manifest(anndata_dir, manifest_pth)
    return manifest_pth


@pytest.fixture(scope="session")
def spatial_build(spatial_manifest, tmp_path_factory) -> SpatialBuild:
    root_tmp_dir = tmp_path_factory.getbasetemp()
    census_dir = root_tmp_dir / "census-builds"
    build_tag = "test-spatial-build"
    blocklist = root_tmp_dir / "block.csv"
    blocklist.touch()

    subprocess.run(
        [
            "coverage",
            "run",
            "--parallel-mode",
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
    obss = []
    for species in ["homo_sapiens", "mus_musculus"]:
        obss.append(census["census_spatial_sequencing"][species].obs.read().concat().to_pandas())
    obs = pd.concat(obss)
    assert set(obs["dataset_id"].unique()) == {d.dataset_id for d in manifest}


def _to_df(somadf: tiledbsoma.DataFrame) -> pd.DataFrame:
    return somadf.read().concat().to_pandas()


def test_locations(spatial_build):
    census = cellxgene_census.open_soma(uri=str(spatial_build.soma_path))
    exp = census["census_spatial_sequencing"]["homo_sapiens"]

    manifest = load_manifest(str(spatial_build.manifest_path), str(spatial_build.blocklist))
    obs = census["census_spatial_sequencing"]["homo_sapiens"].obs.read().concat().to_pandas()
    assay_types = obs[["dataset_id", "assay"]].drop_duplicates().set_index("dataset_id")["assay"]
    h5ads = {d.dataset_id: Path(d.dataset_asset_h5ad_uri) for d in manifest}

    obs_spatial_presence = _to_df(exp["obs_spatial_presence"])

    # Test that the obs_spatial_presence join id + dataset work with locations
    for scene_id, subdf in obs_spatial_presence.groupby("scene_id"):
        assay = assay_types.loc[scene_id]
        sdata = exp.axis_query(
            "RNA", obs_query=tiledbsoma.AxisQuery(value_filter=f"dataset_id == '{scene_id}'")
        ).to_spatialdata(X_name="raw")
        locations_key = list(sdata.shapes.keys())[0]
        locations = sdata.shapes[locations_key]

        assert locations["soma_joinid"].isin(subdf["soma_joinid"]).all()
        assert len(locations["radius"].unique()) == 1

        # Check against values from original file
        if assay == "Visium Spatial Gene Expression":
            with h5py.File(h5ads[scene_id]) as f:
                locations_orig = read_elem(f["obsm/spatial"])
                library_id = list(filter(lambda x: x != "is_single", f["uns/spatial"].keys()))[0]
                radius_orig = f[f"uns/spatial/{library_id}/scalefactors/spot_diameter_fullres"][()] / 2
            np.testing.assert_allclose(
                locations_orig, locations.sort_values("soma_joinid").get_coordinates(), atol=1e-6
            )
            np.testing.assert_allclose(radius_orig, locations["radius"].unique()[0], atol=1e-6)


def test_no_normalized_matrix(spatial_build):
    census = cellxgene_census.open_soma(uri=str(spatial_build.soma_path))
    spatial = census["census_spatial_sequencing"]["homo_sapiens"]

    assert ["raw"] == list(spatial.ms["RNA"]["X"].keys())


def test_images(spatial_build):
    census = cellxgene_census.open_soma(uri=str(spatial_build.soma_path))
    manifest = load_manifest(str(spatial_build.manifest_path), str(spatial_build.blocklist))

    obs = census["census_spatial_sequencing"]["homo_sapiens"].obs.read().concat().to_pandas()
    h5ads = {d.dataset_id: Path(d.dataset_asset_h5ad_uri) for d in manifest}
    assay_types = obs[["dataset_id", "assay"]].drop_duplicates().set_index("dataset_id")["assay"]
    spatial = census["census_spatial_sequencing"]["homo_sapiens"]["spatial"]

    for dataset_id in obs["dataset_id"].unique():
        h5ad_path = h5ads[dataset_id]
        assay = assay_types.loc[dataset_id]
        if assay == "Slide-seqV2":
            # No images should be stored for slide-seq
            assert not hasattr(spatial[dataset_id], "img") or len(spatial[dataset_id].img) == 0
        elif assay == "Visium Spatial Gene Expression":
            with h5py.File(h5ad_path) as f:
                library_id = list(filter(lambda x: x != "is_single", f["uns/spatial"].keys()))[0]
                spatial_dict = read_elem(f[f"uns/spatial/{library_id}"])
            image_collection = spatial[dataset_id]["img"][library_id]
            for k in image_collection.keys():
                from_census = image_collection[k].read().to_numpy()
                from_h5ad = np.transpose(spatial_dict["images"][k], (2, 0, 1))
                np.testing.assert_array_equal(from_h5ad, from_census)


def test_obs_spatial_presence(spatial_build):
    census = cellxgene_census.open_soma(uri=str(spatial_build.soma_path))

    experiment = census["census_spatial_sequencing"]["homo_sapiens"]

    obs = _to_df(experiment.obs)
    obs_spatial_presence = _to_df(experiment["obs_spatial_presence"])

    # For Visium specifically, each observation should show up once and only once in a scene
    assert len(obs_spatial_presence) == len(obs)

    expected = obs[["soma_joinid", "dataset_id"]].rename(columns={"dataset_id": "scene_id"}).assign(data=True)

    # TODO: resolve casting `scene_id` to categorical. Ideally `scene_id` will be stored dictionary encoded
    # https://github.com/single-cell-data/TileDB-SOMA/issues/3743
    pd.testing.assert_frame_equal(expected, obs_spatial_presence.astype({"scene_id": "category"}))


def test_scene(spatial_build):
    census = cellxgene_census.open_soma(uri=str(spatial_build.soma_path))
    experiment = census["census_spatial_sequencing"]["homo_sapiens"]

    for scene in experiment.spatial.values():
        assert isinstance(scene, tiledbsoma.Scene)


def test_spatialdata_query_export(spatial_build):
    census = cellxgene_census.open_soma(uri=str(spatial_build.soma_path))
    experiment = census["census_spatial_sequencing"]["homo_sapiens"]

    obs = experiment["obs"].read().concat().to_pandas()
    joinids = obs["soma_joinid"].iloc[:100].values

    query = experiment.axis_query(
        "RNA",
        obs_query=tiledbsoma.AxisQuery(coords=(joinids,)),
    )
    sdata = query.to_spatialdata(X_name="raw")
    adata = query.to_anndata(X_name="raw")

    pd.testing.assert_frame_equal(
        sdata.tables["RNA"].obs.drop(columns=["instance_key", "region_key"]).reset_index(drop=True),
        adata.obs.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(sdata.tables["RNA"].var, adata.var)
    assert (sdata.tables["RNA"].X != adata.X).sum() == 0
