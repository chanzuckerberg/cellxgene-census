import os
import pathlib
from tempfile import TemporaryDirectory
from typing import List
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import tiledbsoma as soma
from _pytest.monkeypatch import MonkeyPatch

from api.python.cell_census.src import cell_census
from tools.cell_census_builder.__main__ import build, make_experiment_builders
from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.globals import (
    CENSUS_DATA_NAME,
    CENSUS_INFO_NAME,
    CENSUS_X_LAYERS_PLATFORM_CONFIG,
)
from tools.cell_census_builder.mp import process_initializer
from tools.cell_census_builder.util import uricat


class TestBuilder:
    @classmethod
    def setup_class(cls) -> None:
        """
        Setup method that:
        1. Initializes a temporary file system with the correct paths
        2. Patches some configuration to work with fixtures
        3. Calls `process_initializer()` to setup the environment
        """
        process_initializer()

    def test_base_builder_creation(
        self,
        datasets: List[Dataset],
        assets_path: str,
        soma_path: str,
        tmp_path: pathlib.Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """
        Runs the builder, queries the census and performs a set of base assertions.
        """
        monkeypatch.setitem(
            CENSUS_X_LAYERS_PLATFORM_CONFIG["raw"]["tiledb"]["create"]["dims"]["soma_dim_0"], "tile", 2  # type: ignore
        )
        monkeypatch.setitem(
            CENSUS_X_LAYERS_PLATFORM_CONFIG["raw"]["tiledb"]["create"]["dims"]["soma_dim_1"], "tile", 2  # type: ignore
        )
        with patch("tools.cell_census_builder.__main__.prepare_file_system"), patch(
            "tools.cell_census_builder.__main__.build_step1_get_source_assets", return_value=datasets
        ):
            experiment_builders = make_experiment_builders(uricat(soma_path, CENSUS_DATA_NAME), [])  # type: ignore
            from types import SimpleNamespace

            args = SimpleNamespace(multi_process=False, consolidate=False, build_tag="test_tag")
            return_value = build(args, soma_path, assets_path, experiment_builders)  # type: ignore

            # return_value = 0 means that the validation passed
            assert return_value == 0

            # Query the census and do assertions
            census = cell_census.open_soma(uri=soma_path)

            # There are 8 cells in total (4 from the first and 4 from the second datasets). They all belong to homo_sapiens
            human_obs = census[CENSUS_DATA_NAME]["homo_sapiens"]["obs"].read().concat().to_pandas()
            assert human_obs.shape[0] == 8

            # mus_musculus should have 0 cells
            mouse_obs = census[CENSUS_DATA_NAME]["mus_musculus"]["obs"].read().concat().to_pandas()
            assert mouse_obs.shape[0] == 0

            # There are only 4 unique genes
            var = census[CENSUS_DATA_NAME]["homo_sapiens"]["ms"]["RNA"]["var"].read().concat().to_pandas()
            assert var.shape[0] == 4

            # There should be 2 datasets
            returned_datasets = census[CENSUS_INFO_NAME]["datasets"].read().concat().to_pandas()
            assert returned_datasets.shape[0] == 2
            assert list(returned_datasets["dataset_id"]) == ["first_id", "second_id"]

            test_directory = os.listdir(tmp_path)
            assert len(test_directory) == 2
            assert "soma" in test_directory
            assert "h5ads" in test_directory


def test_unicode_support() -> None:
    """
    Regression test that unicode is supported correctly in tiledbsoma.
    This test is not strictly necessary, but it validates the requirements that Cell Census
    support unicode in DataFrame columns.
    """
    with TemporaryDirectory() as d:
        pd_df = pd.DataFrame(data={"value": ["Ünicode", "S̈upport"]}, columns=["value"])
        pd_df["soma_joinid"] = pd_df.index
        s_df = soma.DataFrame(uri=os.path.join(d, "unicode_support")).create(
            pa.Schema.from_pandas(pd_df, preserve_index=False), index_column_names=["soma_joinid"]
        )
        s_df.write(pa.Table.from_pandas(pd_df, preserve_index=False))

        pd_df_in = soma.DataFrame(uri=os.path.join(d, "unicode_support")).read().concat().to_pandas()

        assert pd_df_in["value"].to_list() == ["Ünicode", "S̈upport"]
