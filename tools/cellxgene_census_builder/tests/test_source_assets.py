import pathlib
from types import ModuleType

from cellxgene_census_builder.build_soma.datasets import Dataset
from cellxgene_census_builder.build_soma.source_assets import stage_source_assets
from cellxgene_census_builder.build_state import CensusBuildArgs


def test_source_assets(tmp_path: pathlib.Path, census_build_args: CensusBuildArgs) -> None:
    """
    `source_assets` should copy the datasets from their `dataset_asset_h5ad_uri` to the specified `assets_dir`
    """
    datasets = []
    (tmp_path / "source").mkdir()
    census_build_args.h5ads_path.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        dataset = Dataset(f"dataset_{i}", dataset_asset_h5ad_uri=f"file://{tmp_path}/source/dataset_{i}.h5ad")
        (tmp_path / "source" / f"dataset_{i}.h5ad").touch()
        datasets.append(dataset)

    # Call the function
    stage_source_assets(datasets, census_build_args)

    # Verify that the files exist
    for i in range(10):
        assert (census_build_args.h5ads_path / f"dataset_{i}.h5ad").exists()


def setup_module(module: ModuleType) -> None:
    # this is very important to do early, before any use of `concurrent.futures`
    import multiprocessing

    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)
