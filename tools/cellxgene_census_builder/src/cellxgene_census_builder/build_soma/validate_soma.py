import concurrent.futures
import dataclasses
import gc
import logging
import math
import os.path
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import tiledb
import tiledbsoma as soma
from scipy import sparse
from typing_extensions import Self

from ..build_state import CensusBuildArgs
from ..util import log_process_resource_status, urlcat
from .anndata import make_anndata_cell_filter, open_anndata
from .consolidate import list_uris_to_consolidate
from .datasets import Dataset
from .experiment_builder import ExperimentSpecification
from .experiment_specs import make_experiment_specs
from .globals import (
    CENSUS_DATA_NAME,
    CENSUS_DATASETS_NAME,
    CENSUS_DATASETS_TABLE_SPEC,
    CENSUS_INFO_NAME,
    CENSUS_OBS_STATS_COLUMNS,
    CENSUS_OBS_TABLE_SPEC,
    CENSUS_SCHEMA_VERSION,
    CENSUS_SUMMARY_CELL_COUNTS_NAME,
    CENSUS_SUMMARY_CELL_COUNTS_TABLE_SPEC,
    CENSUS_SUMMARY_NAME,
    CENSUS_VAR_TABLE_SPEC,
    CENSUS_X_LAYERS,
    CXG_OBS_TERM_COLUMNS,
    CXG_SCHEMA_VERSION,
    FEATURE_DATASET_PRESENCE_MATRIX_NAME,
    MEASUREMENT_RNA_NAME,
    SMART_SEQ,
    SOMA_TileDB_Context,
)
from .mp import (
    create_process_pool_executor,
    create_resource_pool_executor,
    log_on_broken_process_pool,
)

logger = logging.getLogger(__name__)


@dataclass  # TODO: use attrs
class EbInfo:
    """Class used to collect information about axis (for validation code)"""

    n_obs: int = 0
    vars: set[str] = dataclasses.field(default_factory=set)
    dataset_ids: set[str] = dataclasses.field(default_factory=set)

    def update(self: Self, b: Self) -> Self:
        self.n_obs += b.n_obs
        self.vars |= b.vars
        self.dataset_ids |= b.dataset_ids
        return self

    @property
    def n_vars(self) -> int:
        return len(self.vars)


def open_experiment(base_uri: str, eb: ExperimentSpecification) -> soma.Experiment:
    """Helper function that knows the Census schema path conventions."""
    return soma.Experiment.open(urlcat(base_uri, CENSUS_DATA_NAME, eb.name), mode="r")


def validate_all_soma_objects_exist(soma_path: str, experiment_specifications: List[ExperimentSpecification]) -> bool:
    """
    Validate all objects present and contain expected metadata.

    soma_path
        +-- census_info: soma.Collection
        |   +-- summary: soma.DataFrame
        |   +-- datasets: soma.DataFrame
        |   +-- summary_cell_counts: soma.DataFrame
        +-- census_data: soma.Collection
        |   +-- homo_sapiens: soma.Experiment
        |   +-- mus_musculus: soma.Experiment
    """

    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        assert soma.Collection.exists(census.uri)
        assert datetime.fromisoformat(census.metadata["created_on"])
        assert "git_commit_sha" in census.metadata

        for name in [CENSUS_INFO_NAME, CENSUS_DATA_NAME]:
            assert soma.Collection.exists(census[name].uri)

        census_info = census[CENSUS_INFO_NAME]
        for name in [CENSUS_DATASETS_NAME, CENSUS_SUMMARY_NAME, CENSUS_SUMMARY_CELL_COUNTS_NAME]:
            assert name in census_info, f"`{name}` missing from census_info"
            assert soma.DataFrame.exists(census_info[name].uri)

        assert sorted(census_info[CENSUS_DATASETS_NAME].keys()) == sorted(CENSUS_DATASETS_TABLE_SPEC.field_names())
        assert sorted(census_info[CENSUS_SUMMARY_CELL_COUNTS_NAME].keys()) == sorted(
            CENSUS_SUMMARY_CELL_COUNTS_TABLE_SPEC.field_names()
        )
        assert sorted(census_info[CENSUS_SUMMARY_NAME].keys()) == sorted(["label", "value", "soma_joinid"])

        census_summary = census[CENSUS_INFO_NAME][CENSUS_SUMMARY_NAME].read().concat().to_pandas()
        assert (
            census_summary.loc[census_summary["label"] == "census_schema_version"].iloc[0]["value"]
            == CENSUS_SCHEMA_VERSION
        )
        assert (
            census_summary.loc[census_summary["label"] == "dataset_schema_version"].iloc[0]["value"]
            == CXG_SCHEMA_VERSION
        )

        # verify required dataset fields are set
        df: pd.DataFrame = census_info[CENSUS_DATASETS_NAME].read().concat().to_pandas()
        assert (df["collection_id"] != "").all()
        assert (df["collection_name"] != "").all()
        assert (df["dataset_title"] != "").all()
        assert (df["dataset_version_id"] != "").all()

        # there should be an experiment for each builder
        census_data = census[CENSUS_DATA_NAME]
        for eb in experiment_specifications:
            assert soma.Experiment.exists(census_data[eb.name].uri)

            e = census_data[eb.name]
            assert soma.DataFrame.exists(e.obs.uri)
            assert soma.Collection.exists(e.ms.uri)

            # there should be a single measurement called 'RNA'
            assert soma.Measurement.exists(e.ms[MEASUREMENT_RNA_NAME].uri)

            # The measurement should contain all X layers where n_obs > 0 (existence checked elsewhere)
            rna = e.ms[MEASUREMENT_RNA_NAME]
            assert soma.DataFrame.exists(rna["var"].uri)
            assert soma.Collection.exists(rna["X"].uri)

            # layers and presence exist only if there are cells in the measurement
            if e.obs.count > 0:
                for lyr in CENSUS_X_LAYERS:
                    assert lyr in rna.X
                    assert soma.SparseNDArray.exists(rna.X[lyr].uri)

                # and a dataset presence matrix
                assert soma.SparseNDArray.exists(rna[FEATURE_DATASET_PRESENCE_MATRIX_NAME].uri)
                assert sum([c.non_zero_length for c in rna["feature_dataset_presence_matrix"].read().coos()]) > 0
                # TODO(atolopko): validate 1) shape, 2) joinids exist in datsets and var

    gc.collect()
    log_process_resource_status()
    return True


def _validate_axis_dataframes(args: Tuple[str, str, Dataset, List[ExperimentSpecification]]) -> Dict[str, EbInfo]:
    assets_path, soma_path, dataset, experiment_specifications = args
    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        census_data = census[CENSUS_DATA_NAME]
        dataset_id = dataset.dataset_id
        unfiltered_ad = open_anndata(assets_path, dataset)
        eb_info: Dict[str, EbInfo] = {}
        for eb in experiment_specifications:
            eb_info[eb.name] = EbInfo()
            anndata_cell_filter = make_anndata_cell_filter(eb.anndata_cell_filter_spec)
            se = census_data[eb.name]
            ad = anndata_cell_filter(unfiltered_ad)
            dataset_obs = (
                se.obs.read(
                    column_names=list(CENSUS_OBS_TABLE_SPEC.field_names()),
                    value_filter=f"dataset_id == '{dataset_id}'",
                )
                .concat()
                .to_pandas()
                .drop(
                    columns=[
                        "dataset_id",
                        "tissue_general",
                        "tissue_general_ontology_term_id",
                        *CENSUS_OBS_STATS_COLUMNS,
                    ]
                )
                .sort_values(by="soma_joinid")
                .drop(columns=["soma_joinid"])
                .reset_index(drop=True)
            )

            # decategorize census obs slice, as it will not have the same categories as H5AD obs,
            # preventing Pandas from performing the DataFrame equivalence operation.
            for key in dataset_obs:
                if isinstance(dataset_obs[key].dtype, pd.CategoricalDtype):
                    dataset_obs[key] = dataset_obs[key].astype(dataset_obs[key].cat.categories.dtype)

            assert len(dataset_obs) == len(ad.obs), f"{dataset.dataset_id}/{eb.name} obs length mismatch"
            if ad.n_obs > 0:
                eb_info[eb.name].n_obs += ad.n_obs
                eb_info[eb.name].dataset_ids.add(dataset_id)
                eb_info[eb.name].vars |= set(ad.var.index.array)
                ad_obs = ad.obs[list(set(CXG_OBS_TERM_COLUMNS) - set(CENSUS_OBS_STATS_COLUMNS))].reset_index(drop=True)
                assert (
                    (dataset_obs.sort_index(axis=1) == ad_obs.sort_index(axis=1)).all().all()
                ), f"{dataset.dataset_id}/{eb.name} obs content, mismatch"

    gc.collect()
    log_process_resource_status()
    return eb_info


def validate_axis_dataframes(
    assets_path: str,
    soma_path: str,
    datasets: List[Dataset],
    experiment_specifications: List[ExperimentSpecification],
    args: CensusBuildArgs,
) -> Dict[str, EbInfo]:
    """ "
    Validate axis dataframes: schema, shape, contents

    Raises on error.  Returns True on success.
    """
    logger.debug("validate_axis_dataframes")
    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        census_data = census[CENSUS_DATA_NAME]

        # check schema
        for eb in experiment_specifications:
            obs = census_data[eb.name].obs
            var = census_data[eb.name].ms[MEASUREMENT_RNA_NAME].var
            assert sorted(obs.keys()) == sorted(CENSUS_OBS_TABLE_SPEC.field_names())
            assert sorted(var.keys()) == sorted(CENSUS_VAR_TABLE_SPEC.field_names())
            for field in obs.schema:
                assert CENSUS_OBS_TABLE_SPEC.field(field.name).is_type_equivalent(
                    field.type
                ), f"Unexpected type in {field.name}: {field.type}"
            for field in var.schema:
                assert CENSUS_VAR_TABLE_SPEC.field(field.name).is_type_equivalent(
                    field.type
                ), f"Unexpected type in {field.name}: {field.type}"

    # check shapes & perform weak test of contents
    eb_info = {eb.name: EbInfo() for eb in experiment_specifications}
    if args.config.multi_process:
        with create_process_pool_executor(args) as ppe:
            futures = [
                ppe.submit(_validate_axis_dataframes, (assets_path, soma_path, dataset, experiment_specifications))
                for dataset in datasets
            ]
            for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                log_on_broken_process_pool(ppe)
                res = future.result()
                for eb_name, ebi in res.items():
                    eb_info[eb_name].update(ebi)
                logger.info(f"validate_axis {n} of {len(datasets)} complete.")
    else:
        for n, dataset in enumerate(datasets, start=1):
            for eb_name, ebi in _validate_axis_dataframes(
                (assets_path, soma_path, dataset, experiment_specifications)
            ).items():
                eb_info[eb_name].update(ebi)
            logger.info(f"validate_axis {n} of {len(datasets)} complete.")

    for eb in experiment_specifications:
        with open_experiment(soma_path, eb) as exp:
            n_vars = len(eb_info[eb.name].vars)

            census_obs_df = exp.obs.read(column_names=["soma_joinid", "dataset_id"]).concat().to_pandas()
            assert eb_info[eb.name].n_obs == len(census_obs_df)
            assert (len(census_obs_df) == 0) or (census_obs_df.soma_joinid.max() + 1 == eb_info[eb.name].n_obs)
            assert eb_info[eb.name].dataset_ids == set(census_obs_df.dataset_id.unique())

            census_var_df = (
                exp.ms[MEASUREMENT_RNA_NAME].var.read(column_names=["feature_id", "soma_joinid"]).concat().to_pandas()
            )
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

    return eb_info


def _validate_X_obs_axis_stats(
    eb: ExperimentSpecification, dataset: Dataset, census_obs: pd.DataFrame, expected_X: sparse.spmatrix
) -> bool:
    """
    Helper function for _validate_X_layers_contents_by_dataset

    Checks that the computed X stats, as stored in obs and var, are correct.
    """
    TypeVar("T", bound=npt.NBitBase)

    def var(X: Union[sparse.csc_matrix, sparse.csr_matrix], axis: int = 0, ddof: int = 1) -> Any:  # cough, cough
        """Helper: variance over sparse matrices"""
        if isinstance(X, np.ndarray):
            return np.var(X, axis=axis, ddof=ddof)

        # Else sparse. Variance of a sparse matrix calculated as
        #   mean(X**2) - mean(X)**2
        # with Bessel's correction applied for unbiased estimate
        X_squared = X.copy()
        X_squared.data **= 2
        n = X.getnnz(axis=axis)
        # catch cases where n<ddof
        with np.errstate(divide="ignore", invalid="ignore"):
            v = ((X_squared.sum(axis=axis).A1 / n) - np.square(X.sum(axis=axis).A1 / n)) * (n / (n - ddof))
            v[~np.isfinite(v)] = 0.0
        return v

    # various datasets have explicit zeros, which are not stored in the Census
    if isinstance(expected_X, (sparse.sparray, sparse.spmatrix)):
        expected_X.eliminate_zeros()

    # obs.raw_sum
    raw_sum = expected_X.sum(axis=1).A1
    assert np.array_equal(
        census_obs.raw_sum.to_numpy(), raw_sum
    ), f"{eb.name}:{dataset.dataset_id} obs.raw_sum incorrect."

    # obs.nnz
    nnz = expected_X.getnnz(axis=1)
    assert np.all(census_obs.nnz.to_numpy() > 0.0)  # All cells must contain at least one count value > 0
    assert np.array_equal(census_obs.nnz.to_numpy(), nnz), f"{eb.name}:{dataset.dataset_id} obs.nnz incorrect."

    # obs.raw_mean_nnz - mean of the explicitly stored values (zeros are _ignored_)
    with np.errstate(divide="ignore"):
        expected_raw_mean_nnz = raw_sum / nnz
    expected_raw_mean_nnz[~np.isfinite(expected_raw_mean_nnz)] = 0.0
    assert np.allclose(
        census_obs.raw_mean_nnz.to_numpy(), expected_raw_mean_nnz
    ), f"{eb.name}:{dataset.dataset_id} obs.raw_mean_nnz incorrect."

    # obs.raw_variance_nnz
    assert np.allclose(
        census_obs.raw_variance_nnz.to_numpy(), var(expected_X, axis=1, ddof=1), rtol=1e-03, atol=1e-05
    ), f"{eb.name}:{dataset.dataset_id} obs.raw_variance_nnz incorrect."

    # obs.n_measured_vars skipped - handled in _validate_Xraw_contents_by_dataset()

    return True


def _validate_Xraw_contents_by_dataset(args: Tuple[str, str, Dataset, List[ExperimentSpecification]]) -> bool:
    """
    Validate that a single dataset is correctly represented in the census. Intended to be
    dispatched from validate_X_layers.

    Currently, implements the following tests:
    * the contents of the X['raw'] matrix are EQUAL for all var feature_ids present in the AnnData
    * the contents of the X['raw'] matrix are EMPTY for all var feature_ids NOT present in the AnnData
    * the contents of the presence matrix match the features present in the AnnData
      (where presence is defined as having a non-zero value)
    """
    assets_path, soma_path, dataset, experiment_specifications = args
    logger.info(f"validate X[raw] by contents - starting {dataset.dataset_id}")
    unfiltered_ad = open_anndata(assets_path, dataset, include_filter_columns=True, var_column_names=("_index",))

    for eb in experiment_specifications:
        with open_experiment(soma_path, eb) as exp:
            anndata_cell_filter = make_anndata_cell_filter(eb.anndata_cell_filter_spec)
            ad = anndata_cell_filter(unfiltered_ad)
            logger.debug(f"AnnData loaded for {eb.name}:{dataset.dataset_id}")

            # get the joinids for the obs axis
            obs_df = (
                exp.obs.read(
                    column_names=["soma_joinid", "dataset_id", *CENSUS_OBS_STATS_COLUMNS],
                    value_filter=f"dataset_id == '{dataset.dataset_id}'",
                )
                .concat()
                .to_pandas()
            )

            assert ad.n_obs == len(obs_df)
            if len(obs_df) == 0:
                continue

            # Assert the stats values look reasonable
            assert all(
                np.isfinite(obs_df[col]).all() and (obs_df[col] >= 0).all()
                for col in ["raw_sum", "nnz", "raw_mean_nnz", "raw_variance_nnz", "n_measured_vars"]
            )

            # get the joinids for the var axis
            var_df = (
                exp.ms[MEASUREMENT_RNA_NAME].var.read(column_names=["soma_joinid", "feature_id"]).concat().to_pandas()
            )
            # mask defines which feature_ids are in the AnnData
            var_joinid_in_adata = var_df.feature_id.isin(ad.var.index)
            assert ad.n_vars == var_joinid_in_adata.sum()

            # var/col reindexer
            var_index = ad.var.join(var_df.set_index("feature_id")).set_index("soma_joinid").index
            var_df = var_df[["soma_joinid"]]  # save some memory

            presence_accumulator = np.zeros((len(var_df),), dtype=np.bool_)

            STRIDE = 125_000
            for idx in range(0, ad.n_obs, STRIDE):
                obs_joinids_split = obs_df.soma_joinid.to_numpy()[idx : idx + STRIDE]
                X_raw = exp.ms[MEASUREMENT_RNA_NAME].X["raw"].read((obs_joinids_split, slice(None))).tables().concat()
                X_raw_data = X_raw["soma_data"].to_numpy()
                X_raw_obs_joinids = X_raw["soma_dim_0"].to_numpy()
                X_raw_var_joinids = X_raw["soma_dim_1"].to_numpy()
                del X_raw

                # positionally re-index
                cols_by_position = var_index.get_indexer(X_raw_var_joinids)  # type: ignore[no-untyped-call]
                rows_by_position = pd.Index(obs_joinids_split).get_indexer(X_raw_obs_joinids)
                del X_raw_obs_joinids

                expected_X = ad[idx : idx + STRIDE].X
                if isinstance(expected_X, np.ndarray):
                    expected_X = sparse.csr_matrix(expected_X)

                # Check that Census summary stats in obs match the AnnData
                assert _validate_X_obs_axis_stats(eb, dataset, obs_df.iloc[idx : idx + STRIDE], expected_X)

                # Check that raw_sum stat matches raw layer stored in the Census
                raw_sum = np.zeros((len(obs_joinids_split),), dtype=np.float64)  # 64 bit for numerical stability
                np.add.at(raw_sum, rows_by_position, X_raw_data)
                raw_sum = raw_sum.astype(
                    CENSUS_OBS_TABLE_SPEC.field("raw_sum").to_pandas_dtype()
                )  # back to the storage type
                assert np.allclose(raw_sum, obs_df.raw_sum.iloc[idx : idx + STRIDE].to_numpy())
                del raw_sum

                # Assertion 1 - the contents of the X matrix are EQUAL for all var values present in the AnnData
                assert (
                    sparse.coo_matrix(
                        (X_raw_data, (rows_by_position, cols_by_position)),
                        shape=(len(obs_joinids_split), ad.shape[1]),
                    )
                    != expected_X
                ).nnz == 0, f"{eb.name}:{dataset.dataset_id} the X matrix elements are not equal."
                del X_raw_data, cols_by_position, rows_by_position, expected_X

                # Assertion 2 - the contents of the X matrix are EMPTY for all var ids NOT present in the AnnData.
                # Test by asserting that no col IDs contain a joinid not in the AnnData.
                assert (
                    var_joinid_in_adata.all()
                    or not pd.Series(X_raw_var_joinids).isin(var_df[~var_joinid_in_adata].soma_joinid).any()
                ), f"{eb.name}:{dataset.dataset_id} unexpected values present in the X matrix."

                presence_accumulator[X_raw_var_joinids] = 1
                del X_raw_var_joinids

                gc.collect()

            # Assertion 3- the contents of the presence matrix match the features present
            # in the AnnData (where presence is defined as having a non-zero value)
            presence = (
                exp.ms[MEASUREMENT_RNA_NAME][FEATURE_DATASET_PRESENCE_MATRIX_NAME]
                .read((dataset.soma_joinid,))
                .tables()
                .concat()
            )

            # sanity check there are no explicit False stored or dups the array
            assert not np.isin(
                presence["soma_data"].to_numpy(), 0
            ).any(), f"{eb.name}:{dataset.dataset_id} unexpected False stored in presence matrix"
            assert (
                np.unique(presence["soma_dim_1"].to_numpy(), return_counts=True)[1] == 1
            ).all(), f"{eb.name}:{dataset.dataset_id} duplicate coordinate in presence matrix"

            mask = np.zeros((exp.ms[MEASUREMENT_RNA_NAME][FEATURE_DATASET_PRESENCE_MATRIX_NAME].shape[1],), dtype=bool)
            mask[presence["soma_dim_1"]] = presence["soma_data"]
            assert np.array_equal(presence_accumulator, mask)
            del mask

            assert (
                obs_df.n_measured_vars.to_numpy() == presence_accumulator.sum()
            ).all(), f"{eb.name}:{dataset.dataset_id} obs.n_measured_vars incorrect."

        # tidy
        del ad, obs_df, var_df, var_index, presence, presence_accumulator, var_joinid_in_adata
        gc.collect()

    del unfiltered_ad
    gc.collect()
    log_process_resource_status()
    logger.info(f"validate X[raw] by contents - finished {dataset.dataset_id}")
    return True


def _validate_X_layer_has_unique_coords(args: Tuple[ExperimentSpecification, str, str, int, int]) -> bool:
    """Validate that all X layers have no duplicate coordinates"""
    experiment_specification, soma_path, layer_name, row_range_start, row_range_stop = args
    with open_experiment(soma_path, experiment_specification) as exp:
        logger.info(
            f"validate_no_dups_X start, {experiment_specification.name}, {layer_name}, rows [{row_range_start}, {row_range_stop})"
        )
        if layer_name not in exp.ms[MEASUREMENT_RNA_NAME].X:
            return True

        X_layer = exp.ms[MEASUREMENT_RNA_NAME].X[layer_name]
        n_rows, n_cols = X_layer.shape
        ROW_SLICE_SIZE = 125_000

        for row in range(row_range_start, min(row_range_stop, n_rows), ROW_SLICE_SIZE):
            slice_of_X = X_layer.read(coords=(slice(row, row + ROW_SLICE_SIZE - 1),)).tables().concat()

            # Use C layout offset for unique test
            offsets = (slice_of_X["soma_dim_0"].to_numpy() * n_cols) + slice_of_X["soma_dim_1"].to_numpy()
            del slice_of_X
            unique_offsets = np.unique(offsets)
            assert len(offsets) == len(unique_offsets)
            del offsets
            gc.collect()

        logger.info(
            f"validate_no_dups_X finished, {experiment_specification.name}, {layer_name}, rows [{row_range_start}, {row_range_stop})"
        )

    gc.collect()
    log_process_resource_status()
    return True


def _validate_Xnorm_layer(args: Tuple[ExperimentSpecification, str, int, int]) -> bool:
    """Validate that X['normalized'] is correct relative to X['raw']"""
    experiment_specification, soma_path, row_range_start, row_range_stop = args
    logger.info(
        f"validate_Xnorm_layer - start, {experiment_specification.name}, rows [{row_range_start}, {row_range_stop})"
    )

    with open_experiment(soma_path, experiment_specification) as exp:
        if "normalized" not in exp.ms[MEASUREMENT_RNA_NAME].X:
            return True

        X_raw = exp.ms[MEASUREMENT_RNA_NAME].X["raw"]
        X_norm = exp.ms[MEASUREMENT_RNA_NAME].X["normalized"]
        assert X_raw.shape == X_norm.shape

        is_smart_seq = np.isin(
            exp.obs.read(column_names=["assay_ontology_term_id"])
            .concat()
            .to_pandas()
            .assay_ontology_term_id.to_numpy(),
            SMART_SEQ,
        )

        var_df = (
            exp.ms[MEASUREMENT_RNA_NAME]
            .var.read(column_names=["soma_joinid", "feature_length"])
            .concat()
            .to_pandas()
            .set_index("soma_joinid")
        )
        n_cols = len(var_df)
        feature_length = var_df.feature_length.to_numpy()
        assert (feature_length > 0).any()
        assert X_raw.shape[1] == n_cols

        ROW_SLICE_SIZE = 25_000
        assert (feature_length.shape[0] * ROW_SLICE_SIZE) < (
            2**31 - 1
        )  # else, will fail in scipy due to int32 overflow during coordinate broadcasting
        for row_idx in range(row_range_start, min(row_range_stop, X_raw.shape[0]), ROW_SLICE_SIZE):
            raw = (
                X_raw.read(coords=(slice(row_idx, row_idx + ROW_SLICE_SIZE - 1),))
                .tables()
                .concat()
                .sort_by([("soma_dim_0", "ascending"), ("soma_dim_1", "ascending")])
            )
            norm = (
                X_norm.read(coords=(slice(row_idx, row_idx + ROW_SLICE_SIZE - 1),))
                .tables()
                .concat()
                .sort_by([("soma_dim_0", "ascending"), ("soma_dim_1", "ascending")])
            )

            assert np.array_equal(raw["soma_dim_0"].to_numpy(), norm["soma_dim_0"].to_numpy())
            assert np.array_equal(raw["soma_dim_1"].to_numpy(), norm["soma_dim_1"].to_numpy())
            # If we wrote a value, it MUST be larger than zero (i.e., represents a raw count value of 1 or greater)
            assert np.all(raw["soma_data"].to_numpy() > 0.0), "Found zero value in raw layer"
            assert np.all(norm["soma_data"].to_numpy() > 0.0), "Found zero value in normalized layer"

            dim0 = norm["soma_dim_0"].to_numpy()
            dim1 = norm["soma_dim_1"].to_numpy()
            row: npt.NDArray[np.int64] = pd.RangeIndex(row_idx, row_idx + ROW_SLICE_SIZE).get_indexer(dim0)  # type: ignore[no-untyped-call]
            col = var_df.index.get_indexer(dim1)
            n_rows: int = row.max() + 1

            norm_csr = sparse.coo_matrix((norm["soma_data"].to_numpy(), (row, col)), shape=(n_rows, n_cols)).tocsr()
            raw_csr = sparse.coo_matrix((raw["soma_data"].to_numpy(), (row, col)), shape=(n_rows, n_cols)).tocsr()
            del raw, norm, dim0, dim1, row, col
            gc.collect()

            sseq_mask = is_smart_seq[row_idx : row_idx + ROW_SLICE_SIZE]
            if sseq_mask.any():
                # this is a very costly operation - do it only when necessary
                raw_csr[sseq_mask, :] /= feature_length
            del sseq_mask

            assert np.allclose(
                norm_csr.sum(axis=1).A1, np.ones((n_rows,), dtype=np.float32), rtol=1e-6, atol=1e-4
            ), f"{experiment_specification.name}: expected normalized X layer to sum to approx 1"
            assert np.allclose(
                norm_csr.data, raw_csr.multiply(1.0 / raw_csr.sum(axis=1).A).tocsr().data, rtol=1e-6, atol=1e-4
            ), f"{experiment_specification.name}: normalized layer does not match raw contents"

            del norm_csr, raw_csr
            gc.collect()

    gc.collect()
    log_process_resource_status()
    logger.info(
        f"validate_Xnorm_layer - finish, {experiment_specification.name}, rows [{row_range_start}, {row_range_stop})"
    )
    return True


def validate_X_layers(
    assets_path: str,
    soma_path: str,
    datasets: List[Dataset],
    experiment_specifications: List[ExperimentSpecification],
    eb_info: Dict[str, EbInfo],
    args: CensusBuildArgs,
) -> bool:
    """ "
    Validate all X layers: schema, shape, contents

    Raises on error.  Returns True on success.
    """
    logger.info("validate_X_layers start")
    avg_row_nnz = 0
    for eb in experiment_specifications:
        with open_experiment(soma_path, eb) as exp:
            assert soma.Collection.exists(exp.ms[MEASUREMENT_RNA_NAME].X.uri)

            census_obs_df = exp.obs.read(column_names=["soma_joinid"]).concat().to_pandas()
            n_obs = len(census_obs_df)
            logger.info(f"uri = {exp.obs.uri}, eb.n_obs = {eb_info[eb.name].n_obs}; n_obs = {n_obs}")
            assert n_obs == eb_info[eb.name].n_obs

            census_var_df = (
                exp.ms[MEASUREMENT_RNA_NAME].var.read(column_names=["feature_id", "soma_joinid"]).concat().to_pandas()
            )
            n_vars = len(census_var_df)
            assert n_vars == eb_info[eb.name].n_vars

            if n_obs > 0:
                for lyr in CENSUS_X_LAYERS:
                    assert soma.SparseNDArray.exists(exp.ms[MEASUREMENT_RNA_NAME].X[lyr].uri)
                    X = exp.ms[MEASUREMENT_RNA_NAME].X[lyr]
                    assert X.schema.field("soma_dim_0").type == pa.int64()
                    assert X.schema.field("soma_dim_1").type == pa.int64()
                    assert X.schema.field("soma_data").type == CENSUS_X_LAYERS[lyr]
                    assert X.shape == (n_obs, n_vars)
                avg_row_nnz = max(avg_row_nnz, math.ceil(exp.ms[MEASUREMENT_RNA_NAME].X["raw"].nnz / n_obs))

    if args.config.multi_process:
        ROWS_PER_PROCESS = 500_000
        mem_budget_factor = int(20 * avg_row_nnz * 2)  # Heuristic:  3 columns, (int64, int64, float32), 100% overhead
        with create_resource_pool_executor(args) as ppe:
            futures = (
                [
                    ppe.submit(
                        100_000 * mem_budget_factor,
                        _validate_Xnorm_layer,
                        (eb, soma_path, row_start, row_start + ROWS_PER_PROCESS),
                    )
                    for eb in experiment_specifications
                    for row_start in range(0, eb_info[eb.name].n_obs, ROWS_PER_PROCESS)
                ]
                + [
                    ppe.submit(
                        400_000 * mem_budget_factor,
                        _validate_X_layer_has_unique_coords,
                        (eb, soma_path, layer_name, row_start, row_start + ROWS_PER_PROCESS),
                    )
                    for eb in experiment_specifications
                    for layer_name in CENSUS_X_LAYERS
                    for row_start in range(0, eb_info[eb.name].n_obs, ROWS_PER_PROCESS)
                ]
                + [
                    ppe.submit(
                        6 * dataset.asset_h5ad_filesize,  # Heuristic value based upon empirical testing.
                        _validate_Xraw_contents_by_dataset,
                        (assets_path, soma_path, dataset, experiment_specifications),
                    )
                    for dataset in datasets
                ]
            )
            for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                log_on_broken_process_pool(ppe)
                assert future.result()
                logger.info(f"validate_X_layers {n} of {len(futures)} complete.")
                log_process_resource_status()

    else:
        for eb in experiment_specifications:
            for layer_name in CENSUS_X_LAYERS:
                logger.info(f"Validating no duplicate coordinates in X layer {eb.name} layer {layer_name}")
                assert _validate_X_layer_has_unique_coords((eb, soma_path, layer_name, 0, n_obs))
        for n, vld in enumerate(
            (
                _validate_Xraw_contents_by_dataset((assets_path, soma_path, dataset, experiment_specifications))
                for dataset in datasets
            ),
            start=1,
        ):
            logger.info(f"validate_X {n} of {len(datasets)} complete.")
            assert vld
        for eb in experiment_specifications:
            assert _validate_Xnorm_layer((eb, soma_path, 0, eb_info[eb.name].n_obs))

    logger.info("validate_X_layers finished")
    return True


def load_datasets_from_census(assets_path: str, soma_path: str) -> List[Dataset]:
    # Datasets are pulled from the census datasets manifest, validating the SOMA
    # census against the snapshot assets.
    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        df = census[CENSUS_INFO_NAME][CENSUS_DATASETS_NAME].read().concat().to_pandas()
        df["dataset_asset_h5ad_uri"] = df.dataset_h5ad_path.map(lambda p: urlcat(assets_path, p))
        assert df.dataset_version_id.is_unique
        assert df.dataset_id.is_unique
        df["asset_h5ad_filesize"] = df.dataset_asset_h5ad_uri.map(lambda p: os.path.getsize(p))
        datasets = Dataset.from_dataframe(df)
        return datasets


def validate_manifest_contents(assets_path: str, datasets: List[Dataset]) -> bool:
    """Confirm contents of manifest are correct."""
    for d in datasets:
        p = pathlib.Path(urlcat(assets_path, d.dataset_h5ad_path))
        assert p.exists() and p.is_file(), f"{d.dataset_h5ad_path} is missing from the census"
        assert str(p).endswith(".h5ad"), "Expected only H5AD assets"

    return True


def validate_consolidation(soma_path: str) -> bool:
    """Verify that obs, var and X layers are all fully consolidated & vacuumed"""

    def is_empty_tiledb_array(uri: str) -> bool:
        with tiledb.open(uri) as A:
            return A.nonempty_domain() is None

    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        consolidated_uris = [obj.uri for obj in list_uris_to_consolidate(census) if obj.is_array()]
        for uri in consolidated_uris:
            # If an empty array, must have fragment count of zero. If a non-empty array,
            # must have fragment count of one.
            assert (len(tiledb.array_fragments(uri)) == 1) or (
                len(tiledb.array_fragments(uri)) == 0 and is_empty_tiledb_array(uri)
            ), f"{uri} has not been fully consolidated & vacuumed"

    return True


def validate_directory_structure(soma_path: str, assets_path: str) -> bool:
    """Verify that the entire census is a single directory tree"""
    assert soma_path.startswith(assets_path.rsplit("/", maxsplit=1)[0])
    assert os.path.exists(soma_path), f"Unable to find SOMA path, expecting {soma_path}"
    assert os.path.exists(assets_path), f"Unable to find assets path, expecting {assets_path}"
    assert soma_path.endswith("soma") and assets_path.endswith("h5ads")
    return True


def validate_relative_path(soma_path: str) -> bool:
    """
    Verify the census objects are stored in the same relative path
    :param soma_path:
    :return:
    """
    # TODO use SOMA API. See https://github.com/single-cell-data/TileDB-SOMA/issues/999

    def _walk_tree(name: str, parent: Any) -> None:
        if isinstance(parent, soma.Collection):
            with tiledb.Group(parent.uri) as parent_group:
                for child in parent_group:
                    assert parent_group.is_relative(child.name), f"{child.name} not relative to {name}"
            for child_name, soma_object in parent.items():
                _walk_tree(".".join([name, child_name]), soma_object)

    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        _walk_tree("census", census)

    return True


def validate_internal_consistency(
    soma_path: str, experiment_specifications: List[ExperimentSpecification], datasets: List[Dataset]
) -> bool:
    """
    Internal checks that various computed stats match.
    """
    logger.info("validate_internal_consistency - cross-checks start")
    datasets_df: pd.DataFrame = Dataset.to_dataframe(datasets).set_index("soma_joinid")

    for eb in experiment_specifications:
        with open_experiment(soma_path, eb) as exp:
            # Load data
            obs = (
                exp.obs.read(column_names=["soma_joinid", "nnz", "dataset_id", "n_measured_vars"]).concat().to_pandas()
            )
            var = (
                exp.ms[MEASUREMENT_RNA_NAME]
                .var.read(column_names=["soma_joinid", "nnz", "n_measured_obs", "feature_id"])
                .concat()
                .to_pandas()
                .set_index("soma_joinid")
            )

            if MEASUREMENT_RNA_NAME in exp.ms and "raw" in exp.ms[MEASUREMENT_RNA_NAME].X:
                presence_tbl = (
                    exp.ms[MEASUREMENT_RNA_NAME][FEATURE_DATASET_PRESENCE_MATRIX_NAME].read().tables().concat()
                )
                presence = sparse.coo_matrix(
                    (
                        presence_tbl["soma_data"],
                        (
                            datasets_df.index.get_indexer(presence_tbl["soma_dim_0"]),  # type: ignore[no-untyped-call]
                            var.index.get_indexer(presence_tbl["soma_dim_1"]),
                        ),
                    ),
                    shape=(len(datasets_df), len(var)),
                ).tocsr()

                # Assertion 1 - counts are mutually consistent
                assert obs.nnz.sum() == var.nnz.sum(), f"{eb.name}: axis NNZ mismatch."
                assert obs.nnz.sum() == exp.ms[MEASUREMENT_RNA_NAME].X["raw"].nnz, f"{eb.name}: axis / X NNZ mismatch."
                assert (
                    exp.ms[MEASUREMENT_RNA_NAME].X["raw"].nnz == exp.ms[MEASUREMENT_RNA_NAME].X["normalized"].nnz
                ), "X['raw'] and X['normalized'] nnz differ."
                assert (
                    exp.ms[MEASUREMENT_RNA_NAME].X["raw"].shape == exp.ms[MEASUREMENT_RNA_NAME].X["normalized"].shape
                ), "X['raw'] and X['normalized'] shape differ."
                assert exp.ms[MEASUREMENT_RNA_NAME].X["raw"].nnz == var.nnz.sum(), "X['raw'] and axis nnz sum differ."

                # Assertion 2 - obs.n_measured_vars is consistent with presence matrix
                """
                approach: sum across presence by dataset. Merge with datasets df on dataset soma_joinid, then
                merge with obs on dataset_id.  Assert that the new column == the n_measured_vars
                """
                datasets_df["presence_sum_var_axis"] = presence.sum(axis=1).A1
                tmp = obs.merge(datasets_df, left_on="dataset_id", right_on="dataset_id")
                assert (
                    tmp.n_measured_vars == tmp.presence_sum_var_axis
                ).all(), f"{eb.name}: obs.n_measured_vars does not match presence matrix."
                del tmp

                # Assertion 3 - var.n_measured_obs is consistent with presence matrix
                tmp = datasets_df.set_index("dataset_id")
                tmp["obs_counts_by_dataset"] = 0
                tmp.update(obs.value_counts(subset="dataset_id").rename("obs_counts_by_dataset"))
                assert (
                    var.n_measured_obs == (tmp.obs_counts_by_dataset.to_numpy() * presence)
                ).all(), f"{eb.name}: var.n_measured_obs does not match presence matrix."
                del tmp

    logger.info("validate_internal_consistency - cross-checks finished")
    return True


def validate_soma_bounding_box(
    soma_path: str, experiment_specifications: List[ExperimentSpecification], eb_info: Dict[str, EbInfo]
) -> bool:
    """
    Verify that single-cell-data/TileDB-SOMA#1969 is not affecting our results.

    Verification is:
        * shape is set correctly
        * no sparse arrays contain the bounding box in metadata
    """

    def get_sparse_arrays(C: soma.Collection) -> List[soma.SparseNDArray]:
        uris = []
        for soma_obj in C.values():
            type = soma_obj.soma_type
            if type == "SOMASparseNDArray":
                uris.append(soma_obj.uri)
            elif type in ["SOMACollection", "SOMAExperiment", "SOMAMeasurement"]:
                uris += get_sparse_arrays(soma_obj)
        return uris

    # first, confirm we set shape correctly, as the code uses it as the max bounding box
    for eb in experiment_specifications:
        with open_experiment(soma_path, eb) as exp:
            n_obs = eb_info[eb.name].n_obs
            n_vars = eb_info[eb.name].n_vars
            for layer_name in exp.ms[MEASUREMENT_RNA_NAME].X:
                assert exp.ms[MEASUREMENT_RNA_NAME].X[layer_name].shape == (n_obs, n_vars)
            if "feature_dataset_presence_matrix" in exp.ms[MEASUREMENT_RNA_NAME]:
                assert exp.ms[MEASUREMENT_RNA_NAME]["feature_dataset_presence_matrix"].shape[1] == n_vars

    with soma.open(soma_path) as C:
        sparse_array_uris = get_sparse_arrays(C)

    # these must not exist
    bbox_metadata_keys = [
        "soma_dim_0_domain_lower",
        "soma_dim_0_domain_upper",
        "soma_dim_1_domain_lower",
        "soma_dim_1_domain_upper",
    ]
    for uri in sparse_array_uris:
        with soma.open(uri) as SA:
            metadata = SA.metadata
            for key in bbox_metadata_keys:
                assert key not in metadata, f"Unexpected bounding box key {key} found in metadata for {uri}"

    return True


def validate(args: CensusBuildArgs) -> bool:
    """
    Validate that the "census" matches the datasets and experiment builder spec.

    Will raise if validation fails. Returns True on success.
    """
    logger.info("Validation start")

    experiment_specifications = make_experiment_specs()

    soma_path = args.soma_path.as_posix()
    assets_path = args.h5ads_path.as_posix()

    assert validate_directory_structure(soma_path, assets_path)

    assert validate_all_soma_objects_exist(soma_path, experiment_specifications)
    assert validate_relative_path(soma_path)
    datasets = load_datasets_from_census(assets_path, soma_path)
    assert validate_manifest_contents(assets_path, datasets)

    assert (eb_info := validate_axis_dataframes(assets_path, soma_path, datasets, experiment_specifications, args))
    assert validate_X_layers(assets_path, soma_path, datasets, experiment_specifications, eb_info, args)
    assert validate_internal_consistency(soma_path, experiment_specifications, datasets)
    if args.config.consolidate:
        assert validate_consolidation(soma_path)
    assert validate_soma_bounding_box(soma_path, experiment_specifications, eb_info)
    logger.info("Validation finished (success)")
    return True
