import pathlib
from typing import List
from unittest.mock import patch

from tools.cell_census_builder.__main__ import build, make_experiment_builders
from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.globals import (
    CENSUS_DATA_NAME,
)
from tools.cell_census_builder.util import uricat
from tools.cell_census_builder.validate import validate


def test_validate(
    datasets: List[Dataset], assets_path: str, soma_path: str, tmp_path: pathlib.Path, setup: None
) -> None:
    """
    Runs the builder, queries the census and performs a set of base assertions.
    """
    with patch("tools.cell_census_builder.__main__.prepare_file_system"), patch(
        "tools.cell_census_builder.__main__.build_step1_get_source_assets", return_value=datasets
    ):
        experiment_builders = make_experiment_builders(uricat(soma_path, CENSUS_DATA_NAME), [])  # type: ignore
        from types import SimpleNamespace

        args = SimpleNamespace(multi_process=False, consolidate=False, build_tag="test_tag")
        return_value = build(args, soma_path, assets_path, experiment_builders)  # type: ignore
        # return_value = 0 means that the build succeeded
        assert return_value == 0

        # validate the cell_census
        return_value = validate(args, soma_path, assets_path, experiment_builders)  # type: ignore
        # return_value = 0 means that the validation passed
        assert return_value is True
