import pathlib
from types import ModuleType, SimpleNamespace

from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.source_assets import stage_source_assets


def test_source_assets(tmp_path: pathlib.Path) -> None:
    """
    `source_assets` should copy the datasets from their `corpora_asset_h5ad_uri` to the specified `assets_dir`
    """
    datasets = []
    pathlib.Path(tmp_path / "source").mkdir()
    pathlib.Path(tmp_path / "dest").mkdir()
    for i in range(10):
        dataset = Dataset(f"dataset_{i}", corpora_asset_h5ad_uri=f"file://{tmp_path}/source/dataset_{i}.h5ad")
        pathlib.Path(tmp_path / "source" / f"dataset_{i}.h5ad").touch()
        datasets.append(dataset)

    # Call the function
    stage_source_assets(datasets, SimpleNamespace(verbose=True), tmp_path / "dest")  # type: ignore

    # Verify that the files exist
    for i in range(10):
        assert pathlib.Path(tmp_path / "dest" / f"dataset_{i}.h5ad").exists()


def setup_module(module: ModuleType) -> None:
    # this is very important to do early, before any use of `concurrent.futures`
    import multiprocessing

    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)
