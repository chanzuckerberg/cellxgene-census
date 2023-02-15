import os
import pathlib
from typing import List
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import tiledb
import tiledbsoma as soma

from tools.cell_census_builder.__main__ import build, make_experiment_builders
from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.globals import (
    CENSUS_DATA_NAME,
    CENSUS_INFO_NAME,
)
from tools.cell_census_builder.util import uricat
from tools.cell_census_builder.validate import validate


def test_base_builder_creation(
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
        assert return_value is True

        # Query the census and do assertions
        census = soma.Collection(
            uri=soma_path,
            context=soma.options.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({"vfs.s3.region": "us-west-2"})),
        )

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


def test_unicode_support(tmp_path: pathlib.Path) -> None:
    """
    Regression test that unicode is supported correctly in tiledbsoma.
    This test is not strictly necessary, but it validates the requirements that Cell Census
    support unicode in DataFrame columns.
    """
    pd_df = pd.DataFrame(data={"value": ["Ünicode", "S̈upport"]}, columns=["value"])
    pd_df["soma_joinid"] = pd_df.index
    s_df = soma.DataFrame(uri=os.path.join(tmp_path, "unicode_support")).create(
        pa.Schema.from_pandas(pd_df, preserve_index=False), index_column_names=["soma_joinid"]
    )
    s_df.write(pa.Table.from_pandas(pd_df, preserve_index=False))

    pd_df_in = soma.DataFrame(uri=os.path.join(tmp_path, "unicode_support")).read().concat().to_pandas()

    assert pd_df_in["value"].to_list() == ["Ünicode", "S̈upport"]
