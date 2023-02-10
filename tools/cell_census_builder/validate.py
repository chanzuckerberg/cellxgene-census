import argparse
import concurrent.futures
import dataclasses
import logging
import os.path
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import tiledbsoma as soma
from scipy import sparse

from .anndata import make_anndata_cell_filter, open_anndata
from .datasets import Dataset
from .experiment_builder import ExperimentBuilder
from .globals import (
    CENSUS_DATA_NAME,
    CENSUS_DATASETS_COLUMNS,
    CENSUS_DATASETS_NAME,
    CENSUS_INFO_NAME,
    CENSUS_OBS_TERM_COLUMNS,
    CENSUS_SCHEMA_VERSION,
    CENSUS_SUMMARY_CELL_COUNTS_COLUMNS,
    CENSUS_SUMMARY_CELL_COUNTS_NAME,
    CENSUS_SUMMARY_NAME,
    CENSUS_VAR_TERM_COLUMNS,
    CENSUS_X_LAYERS,
    CXG_OBS_TERM_COLUMNS,
    CXG_SCHEMA_VERSION,
    FEATURE_DATASET_PRESENCE_MATRIX_NAME,
    SOMA_TileDB_Context,
)
from .mp import create_process_pool_executor
from .util import uricat


@dataclass
class EbInfo:
    """Class used to collect information about axis (for validation code)"""

    n_obs: int = 0
    vars: set[str] = dataclasses.field(default_factory=set)
    dataset_ids: set[str] = dataclasses.field(default_factory=set)

    def update(self: "EbInfo", b: "EbInfo") -> "EbInfo":
        self.n_obs += b.n_obs
        self.vars |= b.vars
        self.dataset_ids |= b.dataset_ids
        return self


def validate_all_soma_objects_exist(soma_path: str, experiment_builders: List[ExperimentBuilder]) -> bool:
    """
    Validate all objects present and contain expected metadata.

    soma_path
        +-- census_info
        |   +-- summary: soma.DataFrame
        |   +-- datasets: soma.DataFrame
        |   +-- summary_cell_counts: soma.DataFrame
        +-- census_data
        |   +-- homo_sapiens: soma.Experiment
        |   +-- mus_musculus: soma.Experiment
    """
    census = soma.Collection(soma_path, context=SOMA_TileDB_Context())
    assert census.exists() and census.soma_type == "SOMACollection"
    assert "cxg_schema_version" in census.metadata and census.metadata["cxg_schema_version"] == CXG_SCHEMA_VERSION
    assert (
        "census_schema_version" in census.metadata and census.metadata["census_schema_version"] == CENSUS_SCHEMA_VERSION
    )
    assert "created_on" in census.metadata and datetime.fromisoformat(census.metadata["created_on"])
    assert "git_commit_sha" in census.metadata

    for name in [CENSUS_INFO_NAME, CENSUS_DATA_NAME]:
        assert name in census
        assert census[name].soma_type == "SOMACollection"
        assert census[name].exists()

    census_info = census[CENSUS_INFO_NAME]
    for name in [CENSUS_DATASETS_NAME, CENSUS_SUMMARY_NAME, CENSUS_SUMMARY_CELL_COUNTS_NAME]:
        assert name in census_info, f"`{name}` missing from census_info"
        assert census_info[name].soma_type == "SOMADataFrame"
        assert census_info[name].exists()

    assert sorted(census_info[CENSUS_DATASETS_NAME].keys()) == sorted(CENSUS_DATASETS_COLUMNS + ["soma_joinid"])
    assert sorted(census_info[CENSUS_SUMMARY_CELL_COUNTS_NAME].keys()) == sorted(
        list(CENSUS_SUMMARY_CELL_COUNTS_COLUMNS) + ["soma_joinid"]
    )
    assert sorted(census_info[CENSUS_SUMMARY_NAME].keys()) == sorted(["label", "value", "soma_joinid"])

    # there should be an experiment for each builder
    census_data = census[CENSUS_DATA_NAME]
    for eb in experiment_builders:
        assert (
            eb.name in census_data
            and census_data[eb.name].exists()
            and census_data[eb.name].soma_type == "SOMAExperiment"
        )

        e = census_data[eb.name]
        assert "obs" in e and e.obs.exists() and e.obs.soma_type == "SOMADataFrame"
        assert "ms" in e and e.ms.exists() and e.ms.soma_type == "SOMACollection"

        # there should be a single measurement called 'RNA'
        assert "RNA" in e.ms and e.ms["RNA"].exists() and e.ms["RNA"].soma_type == "SOMAMeasurement"

        # The measurement should contain all X layers where n_obs > 0 (existence checked elsewhere)
        rna = e.ms["RNA"]
        assert "var" in rna and rna["var"].exists() and rna["var"].soma_type == "SOMADataFrame"
        assert "X" in rna and rna["X"].exists() and rna["X"].soma_type == "SOMACollection"
        for lyr in CENSUS_X_LAYERS:
            # layers only exist if there are cells in the measurement
            if lyr in rna.X:
                assert rna.X[lyr].exists() and rna.X[lyr].soma_type == "SOMASparseNDArray"

        # and a dataset presence matrix
        # dataset presence only exists if there are cells in the measurement
        if FEATURE_DATASET_PRESENCE_MATRIX_NAME in rna:
            assert rna[FEATURE_DATASET_PRESENCE_MATRIX_NAME].exists()
            assert rna[FEATURE_DATASET_PRESENCE_MATRIX_NAME].soma_type == "SOMASparseNDArray"
            # TODO(atolopko): validate 1) shape, 2) joinids exist in datsets and var

    return True


def _validate_axis_dataframes(args: Tuple[str, str, Dataset, List[ExperimentBuilder]]) -> Dict[str, EbInfo]:
    assets_path, soma_path, dataset, experiment_builders = args
    census = soma.Collection(soma_path, context=SOMA_TileDB_Context())
    census_data = census[CENSUS_DATA_NAME]
    dataset_id = dataset.dataset_id
    _, unfiltered_ad = next(open_anndata(assets_path, [dataset], backed="r"))
    eb_info: Dict[str, EbInfo] = {}
    for eb in experiment_builders:
        eb_info[eb.name] = EbInfo()
        anndata_cell_filter = make_anndata_cell_filter(eb.anndata_cell_filter_spec)
        se = census_data[eb.name]
        ad = anndata_cell_filter(unfiltered_ad, retain_X=False)
        dataset_obs = (
            se.obs.read(
                column_names=list(CENSUS_OBS_TERM_COLUMNS),
                value_filter=f"dataset_id == '{dataset_id}'",
            )
            .concat()
            .to_pandas()
            .drop(columns=["dataset_id", "tissue_general", "tissue_general_ontology_term_id"])
            .sort_values(by="soma_joinid")
            .drop(columns=["soma_joinid"])
            .reset_index(drop=True)
        )

        assert len(dataset_obs) == len(ad.obs), f"{dataset.dataset_id}/{eb.name} obs length mismatch"
        if ad.n_obs > 0:
            eb_info[eb.name].n_obs += ad.n_obs
            eb_info[eb.name].dataset_ids.add(dataset_id)
            eb_info[eb.name].vars |= set(ad.var.index.array)
            ad_obs = ad.obs[list(CXG_OBS_TERM_COLUMNS)].reset_index(drop=True)
            assert (dataset_obs == ad_obs).all().all(), f"{dataset.dataset_id}/{eb.name} obs content, mismatch"

    return eb_info


def validate_axis_dataframes(
    assets_path: str,
    soma_path: str,
    datasets: List[Dataset],
    experiment_builders: List[ExperimentBuilder],
    args: argparse.Namespace,
) -> bool:
    """ "
    Validate axis dataframes: schema, shape, contents

    Raises on error.  Returns True on success.
    """
    logging.debug("validate_axis_dataframes")
    census = soma.Collection(soma_path, context=SOMA_TileDB_Context())
    census_data = census[CENSUS_DATA_NAME]

    # check schema
    expected_obs_columns = CENSUS_OBS_TERM_COLUMNS
    expected_var_columns = CENSUS_VAR_TERM_COLUMNS
    for eb in experiment_builders:
        obs = census_data[eb.name].obs
        var = census_data[eb.name].ms["RNA"].var
        assert sorted(obs.keys()) == sorted(expected_obs_columns.keys())
        assert sorted(var.keys()) == sorted(expected_var_columns.keys())
        for field in obs.schema:
            assert field.name in expected_obs_columns
            assert field.type == expected_obs_columns[field.name], f"Unexpected type in {field.name}: {field.type}"
        for field in var.schema:
            assert field.name in expected_var_columns
            assert field.type == expected_var_columns[field.name], f"Unexpected type in {field.name}: {field.type}"

    # check shapes & perform weak test of contents
    eb_info = {eb.name: EbInfo() for eb in experiment_builders}
    if args.multi_process:
        with create_process_pool_executor(args) as ppe:
            futures = [
                ppe.submit(_validate_axis_dataframes, (assets_path, soma_path, dataset, experiment_builders))
                for dataset in datasets
            ]
            for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                res = future.result()
                for eb_name, ebi in res.items():
                    eb_info[eb_name].update(ebi)
                logging.info(f"validate_axis {n} of {len(datasets)} complete.")
    else:
        for n, dataset in enumerate(datasets, start=1):
            for eb_name, ebi in _validate_axis_dataframes(
                (assets_path, soma_path, dataset, experiment_builders)
            ).items():
                eb_info[eb_name].update(ebi)
            logging.info(f"validate_axis {n} of {len(datasets)} complete.")

    for eb in experiment_builders:
        se = census_data[eb.name]
        n_vars = len(eb_info[eb.name].vars)

        census_obs_df = se.obs.read(column_names=["soma_joinid", "dataset_id"]).concat().to_pandas()
        assert eb_info[eb.name].n_obs == len(census_obs_df)
        assert (len(census_obs_df) == 0) or (census_obs_df.soma_joinid.max() + 1 == eb_info[eb.name].n_obs)
        assert eb_info[eb.name].dataset_ids == set(census_obs_df.dataset_id.unique())

        census_var_df = se.ms["RNA"].var.read(column_names=["feature_id", "soma_joinid"]).concat().to_pandas()
        assert n_vars == len(census_var_df)
        assert eb_info[eb.name].vars == set(census_var_df.feature_id.array)
        assert (len(census_var_df) == 0) or (census_var_df.soma_joinid.max() + 1 == n_vars)

        # Validate that all obs soma_joinids are unique and in the range [0, n).
        obs_unique_joinids = np.unique(census_obs_df.soma_joinid.to_numpy())
        assert len(obs_unique_joinids) == len(census_obs_df.soma_joinid.to_numpy())
        assert (len(obs_unique_joinids) == 0) or (
            (obs_unique_joinids[0] == 0) and (obs_unique_joinids[-1] == (len(obs_unique_joinids) - 1))
        )

        # Validate that all var soma_joinids are unique and in the range [0, n).
        var_unique_joinids = np.unique(census_var_df.soma_joinid.to_numpy())
        assert len(var_unique_joinids) == len(census_var_df.soma_joinid.to_numpy())
        assert (len(var_unique_joinids) == 0) or (
            (var_unique_joinids[0] == 0) and var_unique_joinids[-1] == (len(var_unique_joinids) - 1)
        )

    return True


def _validate_X_layers_contents_by_dataset(args: Tuple[str, str, Dataset, List[ExperimentBuilder]]) -> bool:
    """
    Validate that a single dataset is correctly represented in the census.
    Intended to be dispatched from validate_X_layers.

    Currently implements weak tests:
    * the nnz is correct
    * there are no zeros explicitly saved (this is mandated by cell census schema)
    """
    assets_path, soma_path, dataset, experiment_builders = args
    census = soma.Collection(soma_path, context=SOMA_TileDB_Context())
    census_data = census[CENSUS_DATA_NAME]
    _, unfiltered_ad = next(open_anndata(assets_path, [dataset]))
    for eb in experiment_builders:
        se = census_data[eb.name]
        anndata_cell_filter = make_anndata_cell_filter(eb.anndata_cell_filter_spec)
        ad = anndata_cell_filter(unfiltered_ad, retain_X=True)

        soma_joinids: npt.NDArray[np.int64] = (
            se.obs.read(
                column_names=["soma_joinid", "dataset_id"], value_filter=f"dataset_id == '{dataset.dataset_id}'"
            )
            .concat()
            .to_pandas()
            .soma_joinid.to_numpy()
        )

        raw_nnz = 0
        if len(soma_joinids) > 0:
            assert "raw" in se.ms["RNA"].X and se.ms["RNA"].X["raw"].exists()

            def count_elements(arr: soma.SparseNDArray, join_ids: npt.NDArray[np.int64]) -> int:
                # TODO XXX: Work-around for regression TileDB-SOMA#473
                # return sum(t.non_zero_length for t in arr.read((join_ids, slice(None))))
                return sum(t.non_zero_length for t in arr.read((pa.array(join_ids), slice(None))).csrs())

            raw_nnz = count_elements(se.ms["RNA"].X["raw"], soma_joinids)

        def count_nonzero(arr: Union[sparse.spmatrix, npt.NDArray[Any]]) -> int:
            """Return _actual_ non-zero count, NOT the stored value count."""
            if isinstance(arr, (sparse.spmatrix, sparse.coo_array, sparse.csr_array, sparse.csc_array)):
                return np.count_nonzero(arr.data)
            return np.count_nonzero(arr)

        if ad.raw is None:
            assert raw_nnz == count_nonzero(
                ad.X
            ), f"{eb.name}:{dataset.dataset_id} 'raw' nnz mismatch {raw_nnz} vs {count_nonzero(ad.X)}"
        else:
            assert raw_nnz == count_nonzero(
                ad.raw.X
            ), f"{eb.name}:{dataset.dataset_id} 'raw' nnz mismatch {raw_nnz} vs {count_nonzero(ad.raw.X)}"

    return True


def _validate_X_layer_has_unique_coords(args: Tuple[str, ExperimentBuilder, str, int, int]) -> bool:
    """Validate that all X layers have no duplicate coordinates"""
    soma_path, experiment_builder, layer_name, row_range_start, row_range_stop = args
    census = soma.Collection(soma_path, context=SOMA_TileDB_Context())
    census_data = census[CENSUS_DATA_NAME]
    se = census_data[experiment_builder.name]
    logging.info(
        f"validate_no_dups_X start, {experiment_builder.name}, {layer_name}, rows [{row_range_start}, {row_range_stop})"
    )
    if layer_name not in se.ms["RNA"].X:
        return True

    X_layer = se.ms["RNA"].X[layer_name]
    n_rows, n_cols = X_layer.shape
    ROW_SLICE_SIZE = 100_000

    for row in range(row_range_start, min(row_range_stop, n_rows), ROW_SLICE_SIZE):
        # work around TileDB-SOMA bug #900 which errors if we slice beyond end of shape.
        # TODO: remove when issue is resolved.
        end_row = min(row + ROW_SLICE_SIZE, X_layer.shape[0] - 1)

        slice_of_X = X_layer.read(coords=(slice(row, end_row),)).tables().concat()

        # Use C layout offset for unique test
        offsets = (slice_of_X["soma_dim_0"].to_numpy() * n_cols) + slice_of_X["soma_dim_1"].to_numpy()
        unique_offsets = np.unique(offsets)
        assert len(offsets) == len(unique_offsets)

    return True


def validate_X_layers(
    assets_path: str,
    soma_path: str,
    datasets: List[Dataset],
    experiment_builders: List[ExperimentBuilder],
    args: argparse.Namespace,
) -> bool:
    """ "
    Validate all X layers: schema, shape, contents

    Raises on error.  Returns True on success.
    """
    logging.debug("validate_X_layers")
    census = soma.Collection(soma_path, context=SOMA_TileDB_Context())
    census_data = census[CENSUS_DATA_NAME]

    for eb in experiment_builders:
        se = census_data[eb.name]
        assert se.ms["RNA"].X.exists()

        census_obs_df = se.obs.read(column_names=["soma_joinid"]).concat().to_pandas()
        n_obs = len(census_obs_df)
        assert eb.n_obs == n_obs
        census_var_df = se.ms["RNA"].var.read(column_names=["feature_id", "soma_joinid"]).concat().to_pandas()
        n_vars = len(census_var_df)
        assert eb.n_var == n_vars

        if n_obs > 0:
            for lyr in CENSUS_X_LAYERS:
                assert se.ms["RNA"].X[lyr].exists()
                X = se.ms["RNA"].X[lyr]
                assert X.schema.field("soma_dim_0").type == pa.int64()
                assert X.schema.field("soma_dim_1").type == pa.int64()
                assert X.schema.field("soma_data").type == CENSUS_X_LAYERS[lyr]
                assert X.shape == (n_obs, n_vars)

    if args.multi_process:
        with create_process_pool_executor(args) as ppe:
            ROWS_PER_PROCESS = 1_000_000
            dup_coord_futures = [
                ppe.submit(
                    _validate_X_layer_has_unique_coords,
                    (soma_path, eb, layer_name, row_start, row_start + ROWS_PER_PROCESS),
                )
                for eb in experiment_builders
                for layer_name in CENSUS_X_LAYERS
                for row_start in range(0, eb.n_obs, ROWS_PER_PROCESS)
            ]
            per_dataset_futures = [
                ppe.submit(
                    _validate_X_layers_contents_by_dataset, (assets_path, soma_path, dataset, experiment_builders)
                )
                for dataset in datasets
            ]
            for n, future in enumerate(concurrent.futures.as_completed(dup_coord_futures), start=1):
                assert future.result()
                logging.info(f"validate_no_dups_X {n} of {len(dup_coord_futures)} complete.")
            for n, future in enumerate(concurrent.futures.as_completed(per_dataset_futures), start=1):
                assert future.result()
                logging.info(f"validate_X {n} of {len(datasets)} complete.")

    else:
        for eb in experiment_builders:
            for layer_name in CENSUS_X_LAYERS:
                logging.info(f"Validating no duplicate coordinates in X layer {eb.name} layer {layer_name}")
                assert _validate_X_layer_has_unique_coords((soma_path, eb, layer_name, 0, eb.n_obs))
        for n, vld in enumerate(
            (
                _validate_X_layers_contents_by_dataset((assets_path, soma_path, dataset, experiment_builders))
                for dataset in datasets
            ),
            start=1,
        ):
            logging.info(f"validate_X {n} of {len(datasets)} complete.")
            assert vld

    return True


def load_datasets_from_census(assets_path: str, soma_path: str) -> List[Dataset]:
    # Datasets are pulled from the census datasets manifest, validating the SOMA
    # census against the snapshot assets.
    df = soma.Collection(soma_path)[CENSUS_INFO_NAME][CENSUS_DATASETS_NAME].read().concat().to_pandas()
    df.drop(columns=["soma_joinid"], inplace=True)
    df["corpora_asset_h5ad_uri"] = df.dataset_h5ad_path.map(lambda p: uricat(assets_path, p))
    datasets = Dataset.from_dataframe(df)
    return datasets


def validate_manifest_contents(assets_path: str, datasets: List[Dataset]) -> bool:
    """Confirm contents of manifest are correct."""
    for d in datasets:
        p = pathlib.Path(uricat(assets_path, d.dataset_h5ad_path))
        assert p.exists() and p.is_file(), f"{d.dataset_h5ad_path} is missing from the census"
        assert str(p).endswith(".h5ad"), "Expected only H5AD assets"

    return True


def validate(
    args: argparse.Namespace, soma_path: str, assets_path: str, experiment_builders: List[ExperimentBuilder]
) -> bool:
    """
    Validate that the "census" matches the datasets and experiment builder spec.

    Will raise if validation fails. Returns True on success.
    """
    logging.info("Validation start")
    assert soma_path.startswith(assets_path.rsplit("/", maxsplit=1)[0])
    assert os.path.exists(soma_path) and os.path.exists(assets_path)
    assert soma_path.endswith("soma") and assets_path.endswith("h5ads")
    assert validate_all_soma_objects_exist(soma_path, experiment_builders)

    datasets = load_datasets_from_census(assets_path, soma_path)
    assert validate_manifest_contents(assets_path, datasets)

    assert validate_axis_dataframes(assets_path, soma_path, datasets, experiment_builders, args)
    assert validate_X_layers(assets_path, soma_path, datasets, experiment_builders, args)
    logging.info("Validation success")
    return True
