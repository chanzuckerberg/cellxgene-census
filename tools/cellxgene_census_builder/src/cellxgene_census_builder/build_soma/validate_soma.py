from __future__ import annotations

import copy
import dataclasses
import gc
import logging
import os.path
import pathlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self, TypeVar, cast

import dask
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import tiledb
import tiledbsoma as soma
from dask import distributed
from dask.delayed import Delayed
from scipy import sparse

from ..build_state import CensusBuildArgs
from ..logging import logit
from ..util import clamp, cpu_count, log_process_resource_status, urlcat
from .anndata import open_anndata
from .consolidate import list_uris_to_consolidate
from .datasets import Dataset
from .experiment_builder import ExperimentSpecification
from .experiment_specs import make_experiment_specs
from .globals import (
    CENSUS_DATA_NAME,
    CENSUS_DATASETS_NAME,
    CENSUS_DATASETS_TABLE_SPEC,
    CENSUS_INFO_NAME,
    CENSUS_OBS_STATS_FIELDS,
    CENSUS_SCHEMA_VERSION,
    CENSUS_SPATIAL_SEQUENCING_NAME,
    CENSUS_SUMMARY_CELL_COUNTS_NAME,
    CENSUS_SUMMARY_CELL_COUNTS_TABLE_SPEC,
    CENSUS_SUMMARY_NAME,
    CENSUS_VAR_TABLE_SPEC,
    CENSUS_X_LAYERS,
    CXG_SCHEMA_VERSION,
    CXG_VAR_COLUMNS_READ,
    FEATURE_DATASET_PRESENCE_MATRIX_NAME,
    FULL_GENE_ASSAY,
    MEASUREMENT_RNA_NAME,
    SOMA_TileDB_Context,
)
from .mp import create_dask_client, shutdown_dask_cluster

logger = logging.getLogger(__name__)


@dataclass  # TODO: use attrs
class EbInfo:
    """Class used to accumulate information about axis (for validation code)."""

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


def assert_all(__iterable: Iterable[object]) -> bool:
    r = all(__iterable)
    assert r
    return r


def get_census_data_collection_name(eb: ExperimentSpecification) -> str:
    return CENSUS_SPATIAL_SEQUENCING_NAME if eb.is_exclusively_spatial() else CENSUS_DATA_NAME


def get_experiment_unique_key(es: ExperimentSpecification) -> str:
    """Return a unique key for the experiment."""
    return f"{es.name}-{get_census_data_collection_name(es)}"


def get_experiment_uri(base_uri: str, eb: ExperimentSpecification) -> str:
    census_data_collection_name = get_census_data_collection_name(eb)
    return urlcat(base_uri, census_data_collection_name, eb.name)


def open_experiment(base_uri: str, eb: ExperimentSpecification) -> soma.Experiment:
    """Helper function that knows the Census schema path conventions."""
    return soma.Experiment.open(get_experiment_uri(base_uri, eb), mode="r")


def get_experiment_shape(base_uri: str, specs: list[ExperimentSpecification]) -> dict[str, tuple[int, int]]:
    shapes = {}
    for es in specs:
        with open_experiment(base_uri, es) as exp:
            shape = (exp.obs.count, exp.ms[MEASUREMENT_RNA_NAME].var.count)
            shapes[get_experiment_unique_key(es)] = shape
    return shapes


def validate_all_soma_objects_exist(soma_path: str, experiment_specifications: list[ExperimentSpecification]) -> bool:
    """Validate all objects present and contain expected metadata.

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


@logit(logger)
def validate_axis_dataframes_schema(soma_path: str, experiment_specifications: list[ExperimentSpecification]) -> bool:
    """Validate axis dataframe schema matches spec."""
    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        # check schema
        for eb in experiment_specifications:
            census_data = census[eb.root_collection]
            obs = census_data[eb.name].obs
            var = census_data[eb.name].ms[MEASUREMENT_RNA_NAME].var

            assert (
                sorted(obs.keys()) == sorted(eb.obs_table_spec.field_names())
            ), f"Found unexpected fields: {set(obs.keys()).difference(eb.obs_table_spec.field_names())}, and missing fields: {set(eb.obs_table_spec.field_names()).difference(obs.keys())}"
            assert sorted(var.keys()) == sorted(CENSUS_VAR_TABLE_SPEC.field_names())
            for field in obs.schema:
                assert eb.obs_table_spec.field(field.name).is_type_equivalent(
                    field.type
                ), f"Unexpected type in {field.name}: {field.type}"
            for field in var.schema:
                assert CENSUS_VAR_TABLE_SPEC.field(field.name).is_type_equivalent(
                    field.type
                ), f"Unexpected type in {field.name}: {field.type}"

    return True


@logit(logger)
def validate_axis_dataframes_global_ids(
    soma_path: str,
    experiment_specifications: list[ExperimentSpecification],
    eb_info: dict[str, EbInfo],
) -> bool:
    """Validate axes joinid assignment, shape, etc."""
    for eb in experiment_specifications:
        with open_experiment(soma_path, eb) as exp:
            # obs

            census_obs_df = (
                exp.obs.read(
                    column_names=[
                        "soma_joinid",
                        "dataset_id",
                        "tissue_type",
                        "raw_sum",
                        "nnz",
                        "raw_mean_nnz",
                        "raw_variance_nnz",
                        "n_measured_vars",
                    ]
                )
                .concat()
                .to_pandas()
            )
            eb_info_key = get_experiment_uri(soma_path, eb)
            assert eb_info[eb_info_key].n_obs == len(census_obs_df) == exp.obs.count
            assert (len(census_obs_df) == 0) or (census_obs_df.soma_joinid.max() + 1 == eb_info[eb_info_key].n_obs)
            assert eb_info[eb_info_key].dataset_ids == set(census_obs_df.dataset_id.unique())

            # Validate that all obs soma_joinids are unique and in the range [0, n).
            obs_unique_joinids = np.unique(census_obs_df.soma_joinid.to_numpy())
            assert len(obs_unique_joinids) == len(census_obs_df.soma_joinid.to_numpy())
            assert (len(obs_unique_joinids) == 0) or (
                (obs_unique_joinids[0] == 0) and (obs_unique_joinids[-1] == (len(obs_unique_joinids) - 1))
            )

            # Validate that we only contain primary tissue cells and organoids, no cell culture, etc.
            # See census schema for more info.
            assert (census_obs_df.tissue_type.isin(["tissue", "organoid"])).all()

            # Assert the stats values look reasonable
            assert all(
                np.isfinite(census_obs_df[col]).all() and (census_obs_df[col] >= 0).all()
                for col in ["raw_sum", "nnz", "raw_mean_nnz", "raw_variance_nnz", "n_measured_vars"]
            )

            del census_obs_df, obs_unique_joinids

            # var
            n_vars = len(eb_info[eb_info_key].vars)

            census_var_df = (
                exp.ms[MEASUREMENT_RNA_NAME].var.read(column_names=["feature_id", "soma_joinid"]).concat().to_pandas()
            )
            assert n_vars == len(census_var_df) == exp.ms[MEASUREMENT_RNA_NAME].var.count
            assert eb_info[eb_info_key].vars == set(census_var_df.feature_id.array)
            assert (len(census_var_df) == 0) or (census_var_df.soma_joinid.max() + 1 == n_vars)

            # Validate that all var soma_joinids are unique and in the range [0, n).
            var_unique_joinids = np.unique(census_var_df.soma_joinid.to_numpy())
            assert len(var_unique_joinids) == len(census_var_df.soma_joinid.to_numpy())
            assert (len(var_unique_joinids) == 0) or (
                (var_unique_joinids[0] == 0) and var_unique_joinids[-1] == (len(var_unique_joinids) - 1)
            )

            del census_var_df

    return True


def validate_axis_dataframes(
    assets_path: str,
    soma_path: str,
    datasets: list[Dataset],
    experiment_specifications: list[ExperimentSpecification],
    args: CensusBuildArgs,
) -> Delayed[dict[str, EbInfo]]:
    @logit(logger, msg="{0.dataset_id}")
    def _validate_axis_dataframes(
        dataset: Dataset, experiment_specifications: list[ExperimentSpecification], assets_path: str, soma_path: str
    ) -> dict[str, EbInfo]:
        eb_info: dict[str, EbInfo] = {}
        for eb in experiment_specifications:
            with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
                census_data_collection = census[get_census_data_collection_name(eb)]
                dataset_id = dataset.dataset_id
                ad = open_anndata(
                    dataset,
                    base_path=assets_path,
                    obs_column_names=tuple([f.name for f in eb.obs_term_fields_read]),
                    var_column_names=CXG_VAR_COLUMNS_READ,
                    filter_spec=eb.anndata_cell_filter_spec,
                )
                se = census_data_collection[eb.name]

                # NOTE: Since we are validating data for each experiment, we
                # use the experiment uri as the key for the data that must be validated.
                # Using just the experiment spec name would cause collisions as in the case
                # of spatial and non-spatial experiments with the same name (experiment spec name)
                # but stored under different census root collections
                eb_info_key = get_experiment_uri(soma_path, eb)
                eb_info[eb_info_key] = EbInfo()

                dataset_obs = (
                    se.obs.read(
                        column_names=list(eb.obs_table_spec.field_names()),
                        value_filter=f"dataset_id == '{dataset_id}'",
                    )
                    .concat()
                    .to_pandas()
                    .drop(
                        columns=[
                            "dataset_id",
                            "tissue_general",
                            "tissue_general_ontology_term_id",
                            *[x.name for x in CENSUS_OBS_STATS_FIELDS],
                        ]
                    )
                    .sort_values(by="soma_joinid")
                    .drop(columns=["soma_joinid"])
                    .reset_index(drop=True)
                )

                # For spatial assays, make sure we only include primary data
                if eb.is_exclusively_spatial():
                    assert dataset_obs["is_primary_data"].all()

                # decategorize census obs slice, as it will not have the same categories as H5AD obs,
                # preventing Pandas from performing the DataFrame equivalence operation.
                for key in dataset_obs:
                    if isinstance(dataset_obs[key].dtype, pd.CategoricalDtype):
                        dataset_obs[key] = dataset_obs[key].astype(dataset_obs[key].cat.categories.dtype)

                assert (
                    len(dataset_obs) == len(ad.obs)
                ), f"{dataset.dataset_id}/{eb.name} obs length mismatch soma experiment obs len: {len(dataset_obs)} != anndata obs len: {len(ad.obs)}"
                if ad.n_obs > 0:
                    eb_info[eb_info_key].n_obs += ad.n_obs
                    eb_info[eb_info_key].dataset_ids.add(dataset_id)
                    eb_info[eb_info_key].vars |= set(ad.var.index.array)
                    # TODO: Make this comparison work in the case where slide-seq data does not have the visium specific obs columns
                    # Ideally, we fill the ad.obs with a fill value for cases where the column is optional
                    common_cols = ad.obs.columns.intersection(
                        list({f.name for f in eb.obs_term_fields} - {x.name for x in CENSUS_OBS_STATS_FIELDS})
                    )
                    ad_obs = ad.obs[common_cols].reset_index(drop=True)
                    assert (
                        (dataset_obs[common_cols].sort_index(axis=1) == ad_obs.sort_index(axis=1)).all().all()
                    ), f"{dataset.dataset_id}/{eb.name} obs content, mismatch"

        return eb_info

    def reduce_eb_info(results: Sequence[dict[str, EbInfo]]) -> dict[str, EbInfo]:
        eb_info = {}
        for res in results:
            for eb_info_key, info in res.items():
                if eb_info_key not in eb_info:
                    eb_info[eb_info_key] = copy.copy(info)
                else:
                    eb_info[eb_info_key].update(info)
        return eb_info

    eb_info = (
        dask.bag.from_sequence(datasets, partition_size=8)
        .map(
            _validate_axis_dataframes,
            experiment_specifications=experiment_specifications,
            assets_path=assets_path,
            soma_path=soma_path,
        )
        .reduction(reduce_eb_info, reduce_eb_info)
    )

    return eb_info


def validate_X_layers_normalized(
    soma_path: str, experiment_specifications: list[ExperimentSpecification]
) -> Delayed[bool]:
    """Validate that X['normalized'] is correct relative to X['raw']."""

    @logit(logger, msg="{0.name} rows [{1}, {2})")
    def _validate_X_layers_normalized(
        experiment_specification: ExperimentSpecification, row_range_start: int, row_range_stop: int, soma_path: str
    ) -> bool:
        with open_experiment(soma_path, experiment_specification) as exp:
            if "normalized" not in exp.ms[MEASUREMENT_RNA_NAME].X:
                return True

            X_raw = exp.ms[MEASUREMENT_RNA_NAME].X["raw"]
            X_norm = exp.ms[MEASUREMENT_RNA_NAME].X["normalized"]
            assert X_raw.shape == X_norm.shape

            row_range_stop = min(X_raw.shape[0], row_range_stop)

            is_full_gene_assay = np.isin(
                exp.obs.read(
                    coords=(slice(row_range_start, row_range_stop - 1),), column_names=["assay_ontology_term_id"]
                )
                .concat()
                .to_pandas()
                .assay_ontology_term_id.to_numpy(),
                FULL_GENE_ASSAY,
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

            raw = (
                X_raw.read(coords=(slice(row_range_start, row_range_stop - 1),))
                .tables()
                .concat()
                .sort_by([("soma_dim_0", "ascending"), ("soma_dim_1", "ascending")])
            )
            raw_soma_data = raw["soma_data"].to_numpy()
            raw_soma_dim_0 = raw["soma_dim_0"].to_numpy()
            raw_soma_dim_1 = raw["soma_dim_1"].to_numpy()
            del raw

            norm = (
                X_norm.read(coords=(slice(row_range_start, row_range_stop - 1),))
                .tables()
                .concat()
                .sort_by([("soma_dim_0", "ascending"), ("soma_dim_1", "ascending")])
            )
            norm_soma_data = norm["soma_data"].to_numpy()
            norm_soma_dim_0 = norm["soma_dim_0"].to_numpy()
            norm_soma_dim_1 = norm["soma_dim_1"].to_numpy()
            del norm

            # confirm identical coordinates
            assert np.array_equal(raw_soma_dim_0, norm_soma_dim_0)
            assert np.array_equal(raw_soma_dim_1, norm_soma_dim_1)
            del norm_soma_dim_0, norm_soma_dim_1

            # If we wrote a value, it MUST be larger than zero (i.e., represents a raw count value of 1 or greater)
            assert np.all(raw_soma_data > 0.0), "Found zero value in raw layer"
            assert np.all(norm_soma_data > 0.0), "Found zero value in normalized layer"

            row: npt.NDArray[np.int64] = pd.RangeIndex(row_range_start, row_range_stop).get_indexer(raw_soma_dim_0)  # type: ignore[no-untyped-call]
            col = var_df.index.get_indexer(raw_soma_dim_1)
            n_rows: int = row.max() + 1
            del raw_soma_dim_0, raw_soma_dim_1

            norm_csr = sparse.coo_matrix((norm_soma_data, (row, col)), shape=(n_rows, n_cols)).tocsr()
            raw_csr = sparse.coo_matrix((raw_soma_data, (row, col)), shape=(n_rows, n_cols)).tocsr()
            del row, col

            if is_full_gene_assay.any():
                # this is a very costly operation - do it only when necessary
                raw_csr[is_full_gene_assay, :] /= feature_length
            del is_full_gene_assay

            assert np.allclose(
                norm_csr.sum(axis=1).A1, np.ones((n_rows,), dtype=np.float32), rtol=1e-6, atol=1e-4
            ), f"{experiment_specification.name}: expected normalized X layer to sum to approx 1, rows [{row_range_start}, {row_range_stop})"
            assert np.allclose(
                norm_csr.data, raw_csr.multiply(1.0 / raw_csr.sum(axis=1).A).tocsr().data, rtol=1e-6, atol=1e-4
            ), f"{experiment_specification.name}: normalized layer does not match raw contents, rows [{row_range_start}, {row_range_stop})"

            del norm_csr, raw_csr
            gc.collect()

        return True

    JOINID_STRIDE = 32_000
    X_shapes = get_experiment_shape(soma_path, experiment_specifications)
    # JOINID_STRIDE must be smallish, else will fail in scipy due to int32 overflow during coordinate
    # broadcasting. For more info, see: https://github.com/scipy/scipy/issues/13155
    assert all((shape[1] * JOINID_STRIDE < (2**31 - 1)) for shape in X_shapes.values())

    return (
        dask.bag.from_sequence(
            (
                (es, id, id + JOINID_STRIDE)
                for es in experiment_specifications
                for id in range(0, X_shapes[get_experiment_unique_key(es)][0], JOINID_STRIDE)
            ),
            partition_size=8,
        )
        .starmap(_validate_X_layers_normalized, soma_path=soma_path)
        .reduction(all, all)
        .to_delayed()
    )


def validate_X_layers_has_unique_coords(
    soma_path: str, experiment_specifications: list[ExperimentSpecification]
) -> Delayed[bool]:
    """Validate that all X layers have no duplicate coordinates."""

    @logit(logger, msg="{0.name}, {1}, rows [{2}, {3})")
    def _validate_X_layers_has_unique_coords(
        es: ExperimentSpecification,
        layer_name: str,
        row_range_start: int,
        row_range_stop: int,
        soma_path: str,
    ) -> bool:
        with open_experiment(soma_path, es) as exp:
            if layer_name not in exp.ms[MEASUREMENT_RNA_NAME].X:
                return True

            X_layer = exp.ms[MEASUREMENT_RNA_NAME].X[layer_name]
            n_rows, n_cols = X_layer.shape
            slice_of_X = (
                X_layer.read(coords=(slice(row_range_start, min(row_range_stop, n_rows) - 1),)).tables().concat()
            )

            # Use C layout offset for unique test
            offsets = (slice_of_X["soma_dim_0"].to_numpy() * n_cols) + slice_of_X["soma_dim_1"].to_numpy()
            del slice_of_X

            unique_offsets = np.unique(offsets)
            assert len(offsets) == len(unique_offsets)
            del offsets, unique_offsets
            gc.collect()

        return True

    JOINID_STRIDE = 96_000
    X_shapes = get_experiment_shape(soma_path, experiment_specifications)
    return (
        dask.bag.from_sequence(
            (
                (es, layer_name, id, id + JOINID_STRIDE)
                for es in experiment_specifications
                for layer_name in CENSUS_X_LAYERS
                for id in range(0, X_shapes[get_experiment_unique_key(es)][0], JOINID_STRIDE)
            ),
            partition_size=8,
        )
        .starmap(_validate_X_layers_has_unique_coords, soma_path=soma_path)
        .reduction(all, all)
        .to_delayed()
    )


def validate_X_layers_presence(
    soma_path: str, datasets: list[Dataset], experiment_specifications: list[ExperimentSpecification], assets_path: str
) -> Delayed[bool]:
    """Validate that the presence matrix accurately summarizes X[raw] for each experiment.

    Several checks which assume that the contents of X[raw] are correct (assumption verified elsewhere).
    1. No false values explicitly stored in matrix - false are implicit
    2. No duplicate coordinates
    3. Presence mask per dataset is correct for each dataset
    """

    def _read_var_names(path: str) -> npt.NDArray[np.object_]:
        import h5py
        from anndata.io import read_elem

        with h5py.File(path) as f:
            index_key = f["var"].attrs["_index"]
            var_names = read_elem(f["var"][index_key])
            return cast(npt.NDArray[np.object_], var_names)

    @logit(logger)
    def _validate_X_layers_presence_general(experiment_specifications: list[ExperimentSpecification]) -> bool:
        for es in experiment_specifications:
            with open_experiment(soma_path, es) as exp:
                if exp.obs.count == 0:  # skip empty experiments
                    continue

                # Read entire presence matrix for this experiment as COO Table
                presence = exp.ms[MEASUREMENT_RNA_NAME][FEATURE_DATASET_PRESENCE_MATRIX_NAME]
                presence_tbl = presence.read().tables().concat()

                # verify no explicit false stored in presence matrix
                assert not np.isin(
                    presence_tbl["soma_data"].to_numpy(), 0
                ).any(), f"{es.name}: unexpected False stored in presence matrix"

                # Verify no duplicate coords. Use C layout offset for unique test
                offsets = (presence_tbl["soma_dim_0"].to_numpy() * presence.shape[1]) + presence_tbl[
                    "soma_dim_1"
                ].to_numpy()
                unique_offsets = np.unique(offsets)
                assert len(offsets) == len(unique_offsets), f"{es.name}: presence has duplicate coordinates"

        return True

    @logit(logger, msg="{0.dataset_id}")
    def _validate_X_layers_presence(
        dataset: Dataset,
        experiment_specifications: list[ExperimentSpecification],
        soma_path: str,
        assets_path: str,
    ) -> bool:
        """For a given dataset and experiment, confirm that the presence matrix matches contents of X[raw]."""
        for es in experiment_specifications:
            with open_experiment(soma_path, es) as exp:
                obs_df = (
                    exp.obs.read(
                        value_filter=f"dataset_id == '{dataset.dataset_id}'",
                        column_names=["soma_joinid", "n_measured_vars"],
                    )
                    .concat()
                    .to_pandas()
                )
                if len(obs_df) > 0:  # skip empty experiments
                    feature_ids = pd.Index(
                        exp.ms[MEASUREMENT_RNA_NAME]
                        .var.read(column_names=["feature_id"])
                        .concat()
                        .to_pandas()["feature_id"]
                    )

                    presence = (
                        exp.ms[MEASUREMENT_RNA_NAME][FEATURE_DATASET_PRESENCE_MATRIX_NAME]
                        .read((dataset.soma_joinid,))
                        .tables()
                        .concat()
                    )

                    # Get soma_joinids for feature in the original h5ad
                    orig_feature_ids = _read_var_names(f"{assets_path}/{dataset.dataset_h5ad_path}")
                    orig_indices = np.sort(feature_ids.get_indexer(feature_ids.intersection(orig_feature_ids)))

                    np.testing.assert_array_equal(presence["soma_dim_1"], orig_indices)

        return True

    check_presence_values = (
        dask.bag.from_sequence(datasets, partition_size=8)
        .map(
            _validate_X_layers_presence,
            soma_path=soma_path,
            experiment_specifications=experiment_specifications,
            assets_path=assets_path,
        )
        .reduction(all, all)
        .to_delayed()
    )
    return dask.delayed(all)(
        (
            check_presence_values,
            dask.delayed(_validate_X_layers_presence_general)(experiment_specifications),
        )
    )


def validate_X_layers_raw_contents(
    soma_path: str, assets_path: str, datasets: list[Dataset], experiment_specifications: list[ExperimentSpecification]
) -> Delayed[bool]:
    """Validate that X[raw] matches the contents of the filtered H5AD.

    Performs the following tests:
    * the contents of the X['raw'] matrix are EQUAL for all var feature_ids present in the AnnData
    * the contents of the X['raw'] matrix are EMPTY for all var feature_ids NOT present in the AnnData
    """

    def _validate_X_obs_stats(
        eb: ExperimentSpecification, dataset: Dataset, census_obs: pd.DataFrame, expected_X: sparse.spmatrix
    ) -> bool:
        """Helper function for _validate_X_layers_contents_by_dataset.

        Checks that the computed X stats, as stored in obs and var, are correct.
        """
        TypeVar("T", bound=npt.NBitBase)

        def var(X: sparse.csc_matrix | sparse.csr_matrix, axis: int = 0, ddof: int = 1) -> Any:  # cough, cough
            """Helper: variance over sparse matrices."""
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
        if isinstance(expected_X, (sparse.sparray, sparse.spmatrix)):  # noqa: UP038
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

    @logit(logger, msg="{0.dataset_id}")
    def _validate_X_layers_raw_contents(
        dataset: Dataset, experiment_specifications: list[ExperimentSpecification], soma_path: str, assets_path: str
    ) -> bool:
        for es in experiment_specifications:
            ad = open_anndata(
                dataset, base_path=assets_path, filter_spec=es.anndata_cell_filter_spec, var_column_names=("_index",)
            )
            with open_experiment(soma_path, es) as exp:
                # get the joinids for the obs axis
                obs_df = (
                    exp.obs.read(
                        column_names=["soma_joinid", "dataset_id", *[x.name for x in CENSUS_OBS_STATS_FIELDS]],
                        value_filter=f"dataset_id == '{dataset.dataset_id}'",
                    )
                    .concat()
                    .to_pandas()
                )
                assert ad.n_obs == len(obs_df)
                if len(obs_df) == 0:  # skip empty
                    return True

                # get the joinids for the var axis
                var_df = (
                    exp.ms[MEASUREMENT_RNA_NAME]
                    .var.read(column_names=["soma_joinid", "feature_id"])
                    .concat()
                    .to_pandas()
                )
                # mask defines which feature_ids are in the AnnData
                var_joinid_in_adata = var_df.feature_id.isin(ad.var.index)
                assert ad.n_vars == var_joinid_in_adata.sum()

                # var/col reindexer
                var_index = soma.IntIndexer(
                    ad.var.join(var_df.set_index("feature_id")).soma_joinid.to_numpy(), context=exp.context
                )
                var_df = var_df[["soma_joinid"]]  # save some memory

                STRIDE = 48_000  # TODO: scale by density for more consistent memory use
                for idx in range(0, ad.n_obs, STRIDE):
                    obs_joinids_split = obs_df.soma_joinid.to_numpy()[idx : idx + STRIDE]
                    X_raw = (
                        exp.ms[MEASUREMENT_RNA_NAME].X["raw"].read((obs_joinids_split, slice(None))).tables().concat()
                    )
                    X_raw_data = X_raw["soma_data"].to_numpy()
                    X_raw_obs_joinids = X_raw["soma_dim_0"].to_numpy()
                    X_raw_var_joinids = X_raw["soma_dim_1"].to_numpy()
                    del X_raw

                    # positionally re-index
                    cols_by_position = var_index.get_indexer(X_raw_var_joinids)
                    rows_by_position = soma.IntIndexer(obs_joinids_split, context=exp.context).get_indexer(
                        X_raw_obs_joinids
                    )
                    del X_raw_obs_joinids

                    expected_X = ad[idx : idx + STRIDE].X
                    if isinstance(expected_X, np.ndarray):
                        expected_X = sparse.csr_matrix(expected_X)

                    # Check that Census summary stats in obs match the AnnData
                    assert _validate_X_obs_stats(es, dataset, obs_df.iloc[idx : idx + STRIDE], expected_X)

                    # Check that raw_sum stat matches raw layer stored in the Census. This ensures that we have
                    # not accidentally stored _extra_ values, for columns not in the H5AD var dataframe. This
                    # is largely redundant with Assertion #2 - could simply compare raw_sum with the axis sum of
                    # the expected_X matrix.
                    raw_sum = np.zeros((len(obs_joinids_split),), dtype=np.float64)  # 64 bit for numerical stability
                    np.add.at(raw_sum, rows_by_position, X_raw_data)
                    assert np.allclose(
                        raw_sum.astype(
                            es.obs_table_spec.field("raw_sum").to_pandas_dtype()
                        ),  # cast to the storage type
                        obs_df.raw_sum.iloc[idx : idx + STRIDE].to_numpy(),
                    )
                    del raw_sum

                    # Assertion 1 - the contents of the X matrix are EQUAL for all var values present in the AnnData
                    assert (
                        sparse.coo_matrix(
                            (X_raw_data, (rows_by_position, cols_by_position)),
                            shape=(len(obs_joinids_split), ad.shape[1]),
                        )
                        != expected_X
                    ).nnz == 0, f"{es.name}:{dataset.dataset_id} the X matrix elements are not equal."
                    del X_raw_data, cols_by_position, rows_by_position, expected_X

                    # Assertion 2 - the contents of the X matrix are EMPTY for all var ids NOT present in the AnnData.
                    # Test by asserting that no col IDs contain a joinid not in the AnnData.
                    assert (
                        var_joinid_in_adata.all()
                        or not pd.Series(X_raw_var_joinids).isin(var_df[~var_joinid_in_adata].soma_joinid).any()
                    ), f"{es.name}:{dataset.dataset_id} unexpected values present in the X matrix."
                    del X_raw_var_joinids

                    gc.collect()

        return True

    return (
        dask.bag.from_sequence(datasets)
        .map(
            _validate_X_layers_raw_contents,
            experiment_specifications=experiment_specifications,
            soma_path=soma_path,
            assets_path=assets_path,
        )
        .reduction(all, all)
        .to_delayed()
    )


@logit(logger)
def validate_X_layers_schema(
    soma_path: str,
    experiment_specifications: list[ExperimentSpecification],
    eb_info: dict[str, EbInfo],
) -> bool:
    """Validate all X layer schema."""
    for eb in experiment_specifications:
        with open_experiment(soma_path, eb) as exp:
            assert soma.Collection.exists(exp.ms[MEASUREMENT_RNA_NAME].X.uri)

            eb_info_key = get_experiment_uri(soma_path, eb)
            n_obs = eb_info[eb_info_key].n_obs
            n_vars = eb_info[eb_info_key].n_vars
            assert n_obs == exp.obs.count
            assert n_vars == exp.ms[MEASUREMENT_RNA_NAME].var.count

            if n_obs > 0:
                layers = CENSUS_X_LAYERS if not eb.is_exclusively_spatial() else {"raw": pa.float32()}
                for lyr in layers:
                    assert soma.SparseNDArray.exists(exp.ms[MEASUREMENT_RNA_NAME].X[lyr].uri)
                    X = exp.ms[MEASUREMENT_RNA_NAME].X[lyr]
                    assert X.schema.field("soma_dim_0").type == pa.int64()
                    assert X.schema.field("soma_dim_1").type == pa.int64()
                    assert X.schema.field("soma_data").type == CENSUS_X_LAYERS[lyr]
                    assert X.shape == (n_obs, n_vars)

    return True


def load_datasets_from_census(assets_path: str, soma_path: str) -> list[Dataset]:
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


@logit(logger)
def validate_manifest_contents(assets_path: str, datasets: list[Dataset]) -> bool:
    """Confirm contents of manifest are correct."""
    for d in datasets:
        p = pathlib.Path(urlcat(assets_path, d.dataset_h5ad_path))
        assert p.exists() and p.is_file(), f"{d.dataset_h5ad_path} is missing from the census"
        assert str(p).endswith(".h5ad"), "Expected only H5AD assets"

    return True


@logit(logger)
def validate_consolidation(args: CensusBuildArgs) -> bool:
    """Verify that obs, var and X layers are all fully consolidated & vacuumed."""
    if not args.config.consolidate:
        return True

    soma_path = args.soma_path.as_posix()

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


@logit(logger)
def validate_directory_structure(soma_path: str, assets_path: str) -> bool:
    """Verify that the entire census is a single directory tree."""
    assert soma_path.startswith(assets_path.rsplit("/", maxsplit=1)[0])
    assert os.path.exists(soma_path), f"Unable to find SOMA path, expecting {soma_path}"
    assert os.path.exists(assets_path), f"Unable to find assets path, expecting {assets_path}"
    assert soma_path.endswith("soma") and assets_path.endswith("h5ads")
    return True


@logit(logger)
def validate_relative_path(soma_path: str) -> bool:
    """Verify the census objects are stored in the same relative path.

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


@logit(logger)
def validate_internal_consistency(
    soma_path: str, experiment_specifications: list[ExperimentSpecification], datasets: list[Dataset]
) -> bool:
    """Internal checks that various computed stats match."""
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
                del presence_tbl

                # Assertion 1 - counts are mutually consistent
                assert obs.nnz.sum() == var.nnz.sum(), f"{eb.name}: axis NNZ mismatch."
                assert obs.nnz.sum() == exp.ms[MEASUREMENT_RNA_NAME].X["raw"].nnz, f"{eb.name}: axis / X NNZ mismatch."
                if "normalized" in exp.ms[MEASUREMENT_RNA_NAME].X:
                    # Checks on assumptions of normalized layer, which is not present for all experiments
                    assert (
                        exp.ms[MEASUREMENT_RNA_NAME].X["raw"].nnz == exp.ms[MEASUREMENT_RNA_NAME].X["normalized"].nnz
                    ), "X['raw'] and X['normalized'] nnz differ."
                    assert (
                        exp.ms[MEASUREMENT_RNA_NAME].X["raw"].shape
                        == exp.ms[MEASUREMENT_RNA_NAME].X["normalized"].shape
                    ), "X['raw'] and X['normalized'] shape differ."
                assert exp.ms[MEASUREMENT_RNA_NAME].X["raw"].nnz == var.nnz.sum(), "X['raw'] and axis nnz sum differ."

                # Assertion 2 - obs.n_measured_vars is consistent with presence matrix
                """
                approach: sum across presence by dataset. Merge with datasets df on dataset soma_joinid, then
                merge with obs on dataset_id.  Assert that the new column == the n_measured_vars
                """
                datasets_df["presence_sum_var_axis"] = presence.sum(axis=1).A1
                tmp = obs.merge(datasets_df, left_on="dataset_id", right_on="dataset_id")
                try:
                    np.testing.assert_array_equal(
                        tmp["n_measured_vars"],
                        tmp["presence_sum_var_axis"],
                    )
                except AssertionError as e:
                    e.add_note(f"{eb.name}: obs.n_measured_vars does not match presence matrix.")
                    raise
                del tmp

                # Assertion 3 - var.n_measured_obs is consistent with presence matrix
                tmp = datasets_df.set_index("dataset_id")
                tmp["obs_counts_by_dataset"] = 0
                # https://github.com/pandas-dev/pandas/issues/57124
                tmp.update(obs.value_counts(subset="dataset_id").rename("obs_counts_by_dataset"))
                assert (
                    var.n_measured_obs == (tmp.obs_counts_by_dataset.to_numpy() * presence)
                ).all(), f"{eb.name}: var.n_measured_obs does not match presence matrix."
                del tmp

    return True


@logit(logger)
def validate_soma_bounding_box(
    soma_path: str, experiment_specifications: list[ExperimentSpecification], eb_info: dict[str, EbInfo]
) -> bool:
    """Verify that single-cell-data/TileDB-SOMA#1969 is not affecting our results.

    Verification is:
        * shape is set correctly
        * no sparse arrays contain the bounding box in metadata
    """

    def get_sparse_arrays(C: soma.Collection) -> list[soma.SparseNDArray]:
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
            eb_info_key = get_experiment_uri(soma_path, eb)
            n_obs = eb_info[eb_info_key].n_obs
            n_vars = eb_info[eb_info_key].n_vars
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


def validate_soma(args: CensusBuildArgs, client: dask.distributed.Client) -> dask.distributed.Future:
    """Validate that the "census" matches the datasets and experiment builder spec.

    Will raise if validation fails. Returns True on success.
    """
    logger.info("Validation of SOMA objects - start")

    experiment_specifications = make_experiment_specs()

    soma_path = args.soma_path.as_posix()
    assets_path = args.h5ads_path.as_posix()

    assert validate_directory_structure(soma_path, assets_path)
    assert validate_all_soma_objects_exist(soma_path, experiment_specifications)
    assert validate_relative_path(soma_path)
    assert validate_axis_dataframes_schema(soma_path, experiment_specifications)

    datasets = load_datasets_from_census(assets_path, soma_path)
    assert validate_manifest_contents(assets_path, datasets)

    # Scan all H5ADs, check their axis contents, and return a summary.
    eb_info = validate_axis_dataframes(assets_path, soma_path, datasets, experiment_specifications, args)

    # Tasks are scheduled into two priority groups to minimize overall execution time:
    # - long running tasks scheduled higher priority
    # - everything else with default (0) priority
    # This is solely to remove the case of a small number of long-running tasks being
    # delaying overall completion.

    futures: list[dask.distributed.Future] = []

    futures.extend(
        client.compute(
            [
                dask.delayed(assert_all)(
                    (validate_X_layers_raw_contents(soma_path, assets_path, datasets, experiment_specifications),)
                )
            ],
            priority=10,  # higher priority
        )
    )
    futures.extend(
        client.compute(
            [
                dask.delayed(assert_all)(
                    (
                        dask.delayed(lambda e: len(e) == len(experiment_specifications))(eb_info),
                        dask.delayed(validate_axis_dataframes_global_ids)(
                            soma_path, experiment_specifications, eb_info
                        ),
                        dask.delayed(validate_internal_consistency)(soma_path, experiment_specifications, datasets),
                        dask.delayed(validate_soma_bounding_box)(soma_path, experiment_specifications, eb_info),
                        dask.delayed(validate_X_layers_schema)(soma_path, experiment_specifications, eb_info),
                        validate_X_layers_normalized(soma_path, experiment_specifications),
                        validate_X_layers_has_unique_coords(soma_path, experiment_specifications),
                        validate_X_layers_presence(soma_path, datasets, experiment_specifications, assets_path),
                    )
                )
            ],
            priority=0,  # default priority
        )
    )

    return futures


def validate(args: CensusBuildArgs) -> int:
    """Validate all.

    Stand-alone function, to validate outside of a build, for use in CLI
    """
    logger.info("Validating correct consolidation and vacuuming - start")
    n_workers = clamp(cpu_count(), 1, args.config.max_worker_processes)

    with create_dask_client(args, n_workers=n_workers, threads_per_worker=1, memory_limit=None) as client:
        assert all(r.result() for r in distributed.wait(validate_soma(args, client)).done)
        shutdown_dask_cluster(client)
        logger.info("Validation complete.")

    assert validate_consolidation(args)
    logger.info("Validating correct consolidation and vacuuming - complete")
    return 0  # exit code for CLI
