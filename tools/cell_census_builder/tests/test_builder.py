import io
import os
import pathlib
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import tiledb
import tiledbsoma as soma

from tools.cell_census_builder.__main__ import build, build_step1_get_source_datasets, make_experiment_specs
from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.experiment_builder import ExperimentBuilder
from tools.cell_census_builder.globals import (
    CENSUS_DATA_NAME,
    CENSUS_INFO_NAME,
    FEATURE_DATASET_PRESENCE_MATRIX_NAME,
    MEASUREMENT_RNA_NAME,
)
from tools.cell_census_builder.validate import validate


def test_base_builder_creation(
    datasets: List[Dataset], assets_path: str, soma_path: str, tmp_path: pathlib.Path, setup: None
) -> None:
    """
    Runs the builder, queries the census and performs a set of base assertions.
    """
    with patch("tools.cell_census_builder.__main__.prepare_file_system"), patch(
        "tools.cell_census_builder.__main__.build_step1_get_source_datasets", return_value=datasets
    ), patch("tools.cell_census_builder.consolidate._run"), patch(
        "tools.cell_census_builder.validate.validate_consolidation", return_value=True
    ):
        # Patching consolidate_tiledb_object, becuase is uses to much memory to run in github actions.
        experiment_specifications = make_experiment_specs()
        experiment_builders = [ExperimentBuilder(spec) for spec in experiment_specifications]

        from types import SimpleNamespace

        args = SimpleNamespace(multi_process=False, consolidate=True, build_tag="test_tag", max_workers=1, verbose=True)
        return_value = build(args, soma_path, assets_path, experiment_builders)  # type: ignore[arg-type]

        # return_value = 0 means that the build succeeded
        assert return_value == 0

        # validate the cell_census
        return_value = validate(args, soma_path, assets_path, experiment_specifications)  # type: ignore[arg-type]
        assert return_value is True

        # Query the census and do assertions
        with soma.Collection.open(
            uri=soma_path,
            context=soma.options.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({"vfs.s3.region": "us-west-2"})),
        ) as census:
            # There are 8 cells in total (4 from the first and 4 from the second datasets). They all belong to homo_sapiens
            human_obs = census[CENSUS_DATA_NAME]["homo_sapiens"]["obs"].read().concat().to_pandas()
            assert human_obs.shape[0] == 8
            assert list(np.unique(human_obs["dataset_id"])) == ["homo_sapiens_0", "homo_sapiens_1"]

            # mus_musculus should have 8 cells
            mouse_obs = census[CENSUS_DATA_NAME]["mus_musculus"]["obs"].read().concat().to_pandas()
            assert mouse_obs.shape[0] == 8
            assert list(np.unique(mouse_obs["dataset_id"])) == ["mus_musculus_0", "mus_musculus_1"]

            # There are only 4 unique genes
            var = census[CENSUS_DATA_NAME]["homo_sapiens"]["ms"]["RNA"]["var"].read().concat().to_pandas()
            assert var.shape[0] == 4
            assert all(var["feature_id"].str.startswith("homo_sapiens"))

            var = census[CENSUS_DATA_NAME]["mus_musculus"]["ms"]["RNA"]["var"].read().concat().to_pandas()
            assert var.shape[0] == 4
            assert all(var["feature_id"].str.startswith("mus_musculus"))

            # There should be 4 total datasets
            returned_datasets = census[CENSUS_INFO_NAME]["datasets"].read().concat().to_pandas()
            assert returned_datasets.shape[0] == 4
            assert list(returned_datasets["dataset_id"]) == [
                "homo_sapiens_0",
                "homo_sapiens_1",
                "mus_musculus_0",
                "mus_musculus_1",
            ]

            # Presence matrix should exist with the correct dimensions
            for exp_name in ["homo_sapiens", "mus_musculus"]:
                fdpm = census[CENSUS_DATA_NAME][exp_name].ms[MEASUREMENT_RNA_NAME][FEATURE_DATASET_PRESENCE_MATRIX_NAME]
                fdpm_matrix = fdpm.read().coos().concat()

                # The first dimension of the presence matrix should map to the soma_joinids of the returned datasets
                dim_0 = fdpm_matrix.to_scipy().row
                assert all(dim_0 >= 0)
                assert all(dim_0 <= max(returned_datasets.soma_joinid))
                assert fdpm_matrix.shape[0] == max(returned_datasets.soma_joinid) + 1

                # All rows indexed by a Dataframe's soma_joinid that does not belong to the experiment contain all zeros
                dense_pm = fdpm_matrix.to_scipy().todense()
                for i, dataset in returned_datasets.iterrows():
                    if dataset["dataset_id"].startswith(exp_name):
                        assert np.count_nonzero(dense_pm[i]) > 0
                    else:
                        assert np.count_nonzero(dense_pm[i]) == 0

                fdpm_df = fdpm.read().tables().concat().to_pandas()
                n_datasets = fdpm_df["soma_dim_0"].nunique()
                n_features = fdpm_df["soma_dim_1"].nunique()
                assert n_datasets == 2
                assert n_features == 4
                assert fdpm.nnz == 8


def test_unicode_support(tmp_path: pathlib.Path) -> None:
    """
    Regression test that unicode is supported correctly in tiledbsoma.
    This test is not strictly necessary, but it validates the requirements that Cell Census
    support unicode in DataFrame columns.
    """
    pd_df = pd.DataFrame(data={"value": ["Ünicode", "S̈upport"]}, columns=["value"])
    pd_df["soma_joinid"] = pd_df.index
    with soma.DataFrame.create(
        uri=os.path.join(tmp_path, "unicode_support"),
        schema=pa.Schema.from_pandas(pd_df, preserve_index=False),
        index_column_names=["soma_joinid"],
    ) as s_df:
        s_df.write(pa.Table.from_pandas(pd_df, preserve_index=False))

    with soma.DataFrame.open(uri=os.path.join(tmp_path, "unicode_support")) as pd_df_in:
        assert pd_df_in.read().concat().to_pandas()["value"].to_list() == ["Ünicode", "S̈upport"]


def test_build_step1_get_source_datasets(tmp_path: pathlib.Path, manifest_csv: io.TextIOWrapper) -> None:
    import pathlib

    pathlib.Path(tmp_path / "dest").mkdir()
    args = SimpleNamespace(manifest=manifest_csv, test_first_n=None, verbose=True)

    # Call the function
    datasets = build_step1_get_source_datasets(args, f"{tmp_path}/dest")  # type: ignore

    # Verify that 2 datasets are returned
    assert len(datasets) == 2

    # Verify that the datasets have been staged
    assert pathlib.Path(tmp_path / "dest" / "dataset_id_1.h5ad").exists()
    assert pathlib.Path(tmp_path / "dest" / "dataset_id_2.h5ad").exists()
