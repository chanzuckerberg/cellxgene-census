import os
import pathlib
from types import ModuleType
from typing import Any
from unittest.mock import patch

import dask
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pytest
import tiledbsoma as soma

from cellxgene_census_builder.build_soma import build
from cellxgene_census_builder.build_soma.build_soma import build_step1_get_source_datasets
from cellxgene_census_builder.build_soma.datasets import Dataset
from cellxgene_census_builder.build_soma.globals import (
    CENSUS_DATA_NAME,
    CENSUS_INFO_NAME,
    CENSUS_SCHEMA_VERSION,
    CENSUS_SUMMARY_NAME,
    CXG_SCHEMA_VERSION,
    FEATURE_DATASET_PRESENCE_MATRIX_NAME,
    MEASUREMENT_RNA_NAME,
)
from cellxgene_census_builder.build_soma.mp import create_dask_client, shutdown_dask_cluster
from cellxgene_census_builder.build_state import CensusBuildArgs
from cellxgene_census_builder.process_init import process_init


@pytest.mark.parametrize(
    "census_build_args",
    (
        {
            "consolidate": False,
            "build_tag": "test_tag",
            "verbose": 0,
            "max_worker_processes": 1,
        },
        {
            "consolidate": True,
            "build_tag": "test_tag",
            "verbose": 1,
            "max_worker_processes": 1,
        },
    ),
    indirect=True,
)
@pytest.mark.parametrize("validate", (True, False))
def test_base_builder_creation(
    datasets: list[Dataset],
    census_build_args: CensusBuildArgs,
    validate: bool,
    setup: None,
) -> None:
    """
    Runs the builder, queries the census and performs a set of base assertions.
    """

    def proxy_create_dask_client(
        *args: CensusBuildArgs,
        **kwargs: Any,
    ) -> dask.distributed.Client:
        from cellxgene_census_builder.build_soma.mp import create_dask_client

        kwargs["processes"] = False
        kwargs["n_workers"] = 1
        kwargs.pop("threads_per_worker")
        return create_dask_client(*args, **kwargs)

    # proxy psutil.virtual_memory to return 1/2 of the actual memory. There
    # are repeated cases where the test runners OOM, and this helps avoid it.
    memstats = psutil.virtual_memory()
    memstats = memstats._replace(total=int(memstats.total // 2))

    def proxy_psutil_virtual_memory() -> psutil._pslinux.svmem:
        return memstats

    with (
        patch("cellxgene_census_builder.build_soma.build_soma.prepare_file_system"),
        patch("cellxgene_census_builder.build_soma.build_soma.build_step1_get_source_datasets", return_value=datasets),
        patch(
            "cellxgene_census_builder.build_soma.build_soma.create_dask_client", side_effect=proxy_create_dask_client
        ),
        patch("psutil.virtual_memory", side_effect=proxy_psutil_virtual_memory),
    ):
        process_init(census_build_args)
        return_value = build(census_build_args, validate=validate)

        # return_value = 0 means that the build succeeded
        assert return_value == 0

        # Query the census and do assertions
        with soma.Collection.open(
            uri=census_build_args.soma_path.as_posix(),
            context=soma.options.SOMATileDBContext(tiledb_config={"vfs.s3.region": "us-west-2"}),
        ) as census:
            # There are 16 cells in total (4 in each dataset). They all belong to homo_sapiens
            human_obs = census[CENSUS_DATA_NAME]["homo_sapiens"]["obs"].read().concat().to_pandas()
            assert human_obs.shape[0] == 16
            assert list(np.unique(human_obs["dataset_id"])) == [
                "homo_sapiens_0",
                "homo_sapiens_1",
                "homo_sapiens_2",
                "homo_sapiens_3",
            ]

            # mus_musculus should have 16 cells
            mouse_obs = census[CENSUS_DATA_NAME]["mus_musculus"]["obs"].read().concat().to_pandas()
            assert mouse_obs.shape[0] == 16
            assert list(np.unique(mouse_obs["dataset_id"])) == [
                "mus_musculus_0",
                "mus_musculus_1",
                "mus_musculus_2",
                "mus_musculus_3",
            ]

            # Assert number of unique genes
            var = census[CENSUS_DATA_NAME]["homo_sapiens"]["ms"]["RNA"]["var"].read().concat().to_pandas()
            assert var.shape[0] == 5
            assert all(var["feature_id"].str.startswith("homo_sapiens"))

            var = census[CENSUS_DATA_NAME]["mus_musculus"]["ms"]["RNA"]["var"].read().concat().to_pandas()
            assert var.shape[0] == 5
            assert all(var["feature_id"].str.startswith("mus_musculus"))

            # There should be 8 total datasets
            returned_datasets = census[CENSUS_INFO_NAME]["datasets"].read().concat().to_pandas()
            assert returned_datasets.shape[0] == 8
            assert list(returned_datasets["dataset_id"]) == [
                "homo_sapiens_0",
                "homo_sapiens_1",
                "homo_sapiens_2",
                "homo_sapiens_3",
                "mus_musculus_0",
                "mus_musculus_1",
                "mus_musculus_2",
                "mus_musculus_3",
            ]

            # Census summary has the correct metadata
            census_summary = census[CENSUS_INFO_NAME][CENSUS_SUMMARY_NAME].read().concat().to_pandas()
            assert (
                census_summary.loc[census_summary["label"] == "census_schema_version"].iloc[0]["value"]
                == CENSUS_SCHEMA_VERSION
            )
            assert (
                census_summary.loc[census_summary["label"] == "dataset_schema_version"].iloc[0]["value"]
                == CXG_SCHEMA_VERSION
            )

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
                assert n_datasets == 4
                assert n_features == 5
                assert fdpm.nnz == 14


def test_unicode_support(tmp_path: pathlib.Path) -> None:
    """Regression test that unicode is supported correctly in tiledbsoma.
    This test is not strictly necessary, but it validates the requirements that Census
    support unicode in DataFrame columns.
    """
    pd_df = pd.DataFrame(data={"value": ["Ünicode", "S̈upport"]}, columns=["value"])
    pd_df["soma_joinid"] = pd_df.index
    with soma.DataFrame.create(
        uri=os.path.join(tmp_path, "unicode_support"),
        schema=pa.Schema.from_pandas(pd_df, preserve_index=False),
        index_column_names=["soma_joinid"],
        domain=[(pd_df["soma_joinid"].min(), pd_df["soma_joinid"].max())],
    ) as s_df:
        s_df.write(pa.Table.from_pandas(pd_df, preserve_index=False))

    with soma.DataFrame.open(uri=os.path.join(tmp_path, "unicode_support")) as pd_df_in:
        assert pd_df_in.read().concat().to_pandas()["value"].to_list() == ["Ünicode", "S̈upport"]


@pytest.mark.parametrize(
    "census_build_args",
    [{"manifest": True, "verbose": 2, "build_tag": "build_tag", "max_worker_processes": 1}],
    indirect=True,
)
def test_build_step1_get_source_datasets(tmp_path: pathlib.Path, census_build_args: CensusBuildArgs) -> None:
    # prereq for build step 1
    census_build_args.h5ads_path.mkdir(parents=True, exist_ok=True)

    # Call the function
    process_init(census_build_args)
    with create_dask_client(census_build_args, processes=False, memory_limit=0, n_workers=1) as client:
        datasets = build_step1_get_source_datasets(census_build_args)
        shutdown_dask_cluster(client)

    # Verify that 2 datasets are returned
    assert len(datasets) == 2

    # Verify that the datasets have been staged
    assert pathlib.Path(tmp_path / "build_tag" / "h5ads" / "dataset_id_1.h5ad").exists()
    assert pathlib.Path(tmp_path / "build_tag" / "h5ads" / "dataset_id_2.h5ad").exists()


def setup_module(module: ModuleType) -> None:
    # this is very important to do early, before any use of `concurrent.futures`
    import multiprocessing

    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)
