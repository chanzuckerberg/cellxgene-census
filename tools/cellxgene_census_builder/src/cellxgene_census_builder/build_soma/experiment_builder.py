import gc
import itertools
import logging
import math
from collections.abc import Generator, Sequence
from contextlib import ExitStack
from functools import reduce
from typing import Any, Self, TypedDict, cast

import attrs
import dask
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import somacore
import tiledbsoma as soma
from scipy import sparse
from somacore.options import OpenMode

from ..build_state import CensusBuildArgs
from ..logging import logit
from ..util import clamp, log_process_resource_status, urlcat
from .anndata import AnnDataFilterSpec, AnnDataProxy, open_anndata
from .datasets import Dataset
from .globals import (
    CENSUS_OBS_PLATFORM_CONFIG,
    CENSUS_OBS_TABLE_SPEC,
    CENSUS_VAR_PLATFORM_CONFIG,
    CENSUS_VAR_TABLE_SPEC,
    CENSUS_X_LAYERS,
    CENSUS_X_LAYERS_PLATFORM_CONFIG,
    CXG_OBS_COLUMNS_READ,
    CXG_VAR_COLUMNS_READ,
    DONOR_ID_IGNORE,
    FEATURE_DATASET_PRESENCE_MATRIX_NAME,
    FULL_GENE_ASSAY,
    MEASUREMENT_RNA_NAME,
    SOMA_TileDB_Context,
)
from .schema_util import TableSpec
from .stats import get_obs_stats, get_var_stats
from .summary_cell_counts import (
    accumulate_summary_counts,
    init_summary_counts_accumulator,
)
from .util import is_nonnegative_integral

logger = logging.getLogger(__name__)


@attrs.define
class PresenceResult:
    dataset_id: str
    dataset_soma_joinid: int
    eb_name: str
    data: npt.NDArray[np.bool_]
    cols: npt.NDArray[np.int64]


@attrs.define
class AxisStats:
    # obs/var stats computed (example: raw_mean_nnz)
    eb_name: str
    obs_stats: pd.DataFrame
    var_stats: pd.DataFrame


AccumulateXResult = tuple[PresenceResult, AxisStats]
AccumulateXResults = Sequence[AccumulateXResult]


def _assert_open_for_write(obj: somacore.SOMAObject | None) -> None:
    assert obj is not None
    assert obj.exists(obj.uri)
    assert obj.mode == "w"
    assert not obj.closed


@attrs.define(frozen=True)
class ExperimentSpecification:
    """Declarative "specification" of a SOMA experiment. This is a read-only
    specification, independent of the datasets used to build the census.

    Parameters:
    * experiment "name" (eg, 'homo_sapiens'), must be unique in all experiments.
    * a human-readable label, e.g, "Homo sapiens"
    * ontology ID
    * an AnnData filter used to cherry pick data for the experiment
    * external reference data used to build the experiment, e.g., gene length data

    Usage: to create, use the factory method `ExperimentSpecification.create(...)`
    """

    name: str
    label: str
    anndata_cell_filter_spec: AnnDataFilterSpec
    organism_ontology_term_id: str

    @classmethod
    def create(
        cls,
        name: str,
        label: str,
        anndata_cell_filter_spec: AnnDataFilterSpec,
        organism_ontology_term_id: str,
    ) -> Self:
        """Factory method. Do not instantiate the class directly."""
        return cls(name, label, anndata_cell_filter_spec, organism_ontology_term_id)


class ExperimentBuilder:
    """Class that embodies the operators and state to build an Experiment.
    The creation and driving of these objects is done by the main loop.
    """

    def __init__(self, specification: ExperimentSpecification):
        self.specification = specification

        # accumulated state
        self.n_obs: int = 0
        self.n_unique_obs: int = 0
        self.n_var: int = 0
        self.n_datasets: int = 0
        self.n_donors: int = 0  # Caution: defined as (unique dataset_id, donor_id) tuples, *excluding* some values
        self.obs_df: pd.DataFrame | None = None
        self.var_df: pd.DataFrame | None = None
        self.dataset_obs_joinid_start: dict[str, int] = {}  # starting joinid per dataset_id
        self.dataset_n_obs: dict[str, int] = {}  # n_obs per dataset_id
        self.census_summary_cell_counts: pd.DataFrame = init_summary_counts_accumulator()
        self.experiment: soma.Experiment | None = None  # initialized in create()
        self.experiment_uri: str | None = None  # initialized in create()
        self.global_var_joinids: pd.DataFrame | None = None
        self.presence: dict[int, tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]] = {}

    @property
    def name(self) -> str:
        return self.specification.name

    @property
    def anndata_cell_filter_spec(self) -> AnnDataFilterSpec:
        return self.specification.anndata_cell_filter_spec

    def create(self, census_data: soma.Collection) -> None:
        """Create experiment within the specified Collection with a single Measurement."""
        logger.info(f"{self.name}: create experiment at {urlcat(census_data.uri, self.name)}")

        self.experiment = census_data.add_new_collection(self.name, soma.Experiment)
        self.experiment_uri = self.experiment.uri

        # create `ms`
        ms = self.experiment.add_new_collection("ms")

        # make measurement and add to ms collection
        ms.add_new_collection(MEASUREMENT_RNA_NAME, soma.Measurement)

    def write_obs_dataframe(self) -> None:
        logger.info(f"{self.name}: writing obs dataframe")
        assert self.experiment is not None
        _assert_open_for_write(self.experiment)

        obs_df = cast(pd.DataFrame, CENSUS_OBS_TABLE_SPEC.recategoricalize(self.obs_df))
        obs_schema = CENSUS_OBS_TABLE_SPEC.to_arrow_schema(obs_df)

        # create `obs`
        self.experiment.add_new_dataframe(
            "obs",
            schema=obs_schema,
            index_column_names=["soma_joinid"],
            platform_config=CENSUS_OBS_PLATFORM_CONFIG,
            domain=[(obs_df["soma_joinid"].min(), obs_df["soma_joinid"].max())],
        )

        if obs_df is None or obs_df.empty:
            logger.info(f"{self.name}: empty obs dataframe")
        else:
            logger.debug(f"experiment {self.name} obs = {obs_df.shape}")
            assert not np.isnan(obs_df.nnz.to_numpy()).any()  # sanity check
            pa_table = pa.Table.from_pandas(obs_df, preserve_index=False, schema=obs_schema)
            self.experiment.obs.write(pa_table)

    def write_var_dataframe(self) -> None:
        logger.info(f"{self.name}: writing var dataframe")
        assert self.experiment is not None
        _assert_open_for_write(self.experiment)

        rna_measurement = self.experiment.ms[MEASUREMENT_RNA_NAME]

        var_df = cast(pd.DataFrame, CENSUS_VAR_TABLE_SPEC.recategoricalize(self.var_df))
        var_schema = CENSUS_VAR_TABLE_SPEC.to_arrow_schema(var_df)

        # create `var` in the measurement
        rna_measurement.add_new_dataframe(
            "var",
            schema=var_schema,
            index_column_names=["soma_joinid"],
            platform_config=CENSUS_VAR_PLATFORM_CONFIG,
            domain=[(var_df["soma_joinid"].min(), var_df["soma_joinid"].max())],
        )

        if var_df is None or var_df.empty:
            logger.info(f"{self.name}: empty var dataframe")
        else:
            logger.debug(f"experiment {self.name} var = {var_df.shape}")
            pa_table = pa.Table.from_pandas(var_df, preserve_index=False, schema=var_schema)
            rna_measurement.var.write(pa_table)

    def create_X_with_layers(self) -> None:
        """Create layers in ms['RNA']/X."""
        logger.info(f"{self.name}: create X layers")

        rna_measurement = self.experiment.ms[MEASUREMENT_RNA_NAME]  # type:ignore
        _assert_open_for_write(rna_measurement)

        # make the `X` collection
        rna_measurement.add_new_collection("X")

        # SOMA does not currently support empty arrays, so special case this corner-case.
        if self.n_obs > 0:
            assert self.n_var > 0
            for layer_name in CENSUS_X_LAYERS:
                rna_measurement["X"].add_new_sparse_ndarray(
                    layer_name,
                    type=CENSUS_X_LAYERS[layer_name],
                    shape=(self.n_obs, self.n_var),
                    platform_config=CENSUS_X_LAYERS_PLATFORM_CONFIG[layer_name],
                )

    def populate_presence_matrix(self, datasets: list[Dataset]) -> None:
        """Save presence matrix per Experiment."""
        _assert_open_for_write(self.experiment)
        logger.info(f"Save presence matrix for {self.name} - start")

        # SOMA does not currently support arrays with a zero length domain, so special case this corner-case
        # where no data has been read for this experiment.
        if len(self.presence) > 0:
            # sanity check
            assert len(self.presence) == self.n_datasets

            max_dataset_joinid = max(d.soma_joinid for d in datasets)

            # LIL is fast way to create spmatrix
            pm = sparse.lil_matrix((max_dataset_joinid + 1, self.n_var), dtype=bool)
            for dataset_joinid, presence in self.presence.items():
                data, cols = presence
                pm[dataset_joinid, cols] = data

            pm = pm.tocoo()
            pm.eliminate_zeros()
            assert pm.count_nonzero() == pm.nnz
            assert pm.dtype == bool

            fdpm = self.experiment.ms[MEASUREMENT_RNA_NAME].add_new_sparse_ndarray(  # type:ignore
                FEATURE_DATASET_PRESENCE_MATRIX_NAME,
                type=pa.bool_(),
                shape=(max_dataset_joinid + 1, self.n_var),
            )
            fdpm.write(pa.SparseCOOTensor.from_scipy(pm))

        logger.info(f"Save presence matrix for {self.name} - finish")
        log_process_resource_status()


def accumulate_axes_dataframes(
    base_path: str,
    datasets: list[Dataset],
    experiment_builders: list[ExperimentBuilder],
) -> list[tuple[ExperimentBuilder, tuple[pd.DataFrame, pd.DataFrame]]]:
    """Two parallel operations.

    From all datasets:
    1. Concat all obs dataframes
    2. Union all var dataframes
    """

    def get_obs_and_var(
        dataset: Dataset, spec: ExperimentSpecification, base_path: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        with open_anndata(
            dataset,
            base_path=base_path,
            filter_spec=spec.anndata_cell_filter_spec,
            obs_column_names=CXG_OBS_COLUMNS_READ,
            var_column_names=CXG_VAR_COLUMNS_READ,
        ) as adata:
            logger.debug(f"{dataset.dataset_id}/{spec.name} - found {adata.n_obs} cells")

            # Skip this dataset if there are not cells after filtering
            if adata.n_obs == 0:
                logger.debug(f"{spec.name} - H5AD has no data after filtering, skipping {dataset.dataset_id}")
                return pd.DataFrame(), pd.DataFrame()

            obs_df = adata.obs.copy()
            obs_df["dataset_id"] = dataset.dataset_id

            var_df = (
                adata.var.copy()
                .rename_axis("feature_id")
                .reset_index()[["feature_id", "feature_name", "feature_length"]]
            )

            return obs_df, var_df

    datasets_bag = dask.bag.from_sequence(datasets)
    df_pairs_per_eb: list[list[tuple[pd.DataFrame, pd.DataFrame]]] = dask.compute(
        *[
            datasets_bag.map(
                get_obs_and_var,
                spec=eb.specification,
                base_path=base_path,
            )
            for eb in experiment_builders
        ]
    )

    return list(
        zip(
            experiment_builders,
            [
                (
                    pd.concat(cast(list[pd.DataFrame], [df_pair[0] for df_pair in df_pairs]), ignore_index=True),
                    pd.concat(
                        cast(list[pd.DataFrame], [df_pair[1] for df_pair in df_pairs]), ignore_index=True
                    ).drop_duplicates(ignore_index=True),
                )
                for df_pairs in df_pairs_per_eb
            ],
            strict=False,
        )
    )


def post_acc_axes_processing(accumulated: list[tuple[ExperimentBuilder, tuple[pd.DataFrame, pd.DataFrame]]]) -> None:
    """Processing steps post-accumulation of all axes dataframes.

    Includes:
    * assign soma_joinids
    * add derived or summary columns
    * generate summary and/or working data for the experiment_builder
    """

    def add_placeholder_columns(df: pd.DataFrame, table_spec: TableSpec, default: dict[npt.DTypeLike, Any]) -> None:
        for key in table_spec.field_names():
            if key not in df:
                dtype = table_spec.field(key).to_pandas_dtype(ignore_dict_type=True)
                fill_value = default[dtype]
                df[key] = np.full((len(df),), fill_value, dtype=dtype)

    def per_dataset_summary_counts(eb: ExperimentBuilder, obs: pd.DataFrame) -> None:
        for _, obs_slice in obs.groupby("dataset_id"):
            assert obs_slice.soma_joinid.max() - obs_slice.soma_joinid.min() + 1 == len(obs_slice)
            eb.census_summary_cell_counts = accumulate_summary_counts(eb.census_summary_cell_counts, obs_slice)

    for eb, (obs, var) in accumulated:
        if not len(obs):
            eb.obs_df = None
            eb.var_df = None
            eb.n_obs = 0
            eb.n_var = 0
            continue

        obs["soma_joinid"] = range(0, len(obs))
        var["soma_joinid"] = range(0, len(var))

        add_tissue_mapping(obs)  # add tissue mapping (e.g., tissue_general term)

        # add columns to be completed later, e.g., summary stats such as mean of X
        add_placeholder_columns(
            obs, CENSUS_OBS_TABLE_SPEC, default={np.int64: np.iinfo(np.int64).min, np.float64: np.nan}
        )
        add_placeholder_columns(var, CENSUS_VAR_TABLE_SPEC, default={np.int64: 0})

        # compute intermediate values used later in the build
        eb.n_datasets = obs.dataset_id.nunique()
        eb.n_unique_obs = (obs.is_primary_data == True).sum()  # noqa: E712
        eb.n_donors = obs[~obs.donor_id.isin(DONOR_ID_IGNORE)].groupby("dataset_id").donor_id.nunique().sum()
        eb.global_var_joinids = var[["feature_id", "soma_joinid"]].set_index("feature_id")

        grouped_by_id = obs.groupby("dataset_id").soma_joinid.agg(["min", "count"])
        eb.dataset_obs_joinid_start = grouped_by_id["min"].to_dict()
        eb.dataset_n_obs = grouped_by_id["count"].to_dict()

        # gather per-dataset summary statistics e.g., cell type counts, etc.
        per_dataset_summary_counts(eb, obs)

        # save results in the ExperimentBuilder
        eb.obs_df = obs
        eb.var_df = var
        eb.n_obs = len(obs)
        eb.n_var = len(var)


def _get_axis_stats(
    raw_X: sparse.spmatrix | npt.NDArray[np.float32],
    dataset_obs_joinid_start: int,
    local_var_joinids: npt.NDArray[np.int64],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate obs and var summary stats, e.g., raw_sum, etc.

    Return tuple of (obs_stats_df, var_stats_df), both indexed by soma_joinid.
    """
    assert sparse.isspmatrix_csr(raw_X) or sparse.isspmatrix_csc(raw_X) or isinstance(raw_X, np.ndarray)

    obs_stats = get_obs_stats(raw_X)
    obs_stats = obs_stats.set_index(pd.RangeIndex(dataset_obs_joinid_start, dataset_obs_joinid_start + len(obs_stats)))
    obs_stats.index.name = "soma_joinid"

    var_stats = get_var_stats(raw_X)
    var_stats = var_stats.set_index(local_var_joinids)
    var_stats.index.name = "soma_joinid"

    return (obs_stats, var_stats)


class XReduction(TypedDict):
    """Information accumulated/reduced from each AnnData X read."""

    dataset_id: str
    obs_stats: pd.DataFrame
    var_stats: pd.DataFrame
    presence: list[PresenceResult]


def reduce_X_stats_chunk(results: Sequence[XReduction]) -> XReduction:
    """Reduce multiple XReduction objects into one."""
    results = list(results)
    assert len(results)
    if len(results) == 1:
        return results[0].copy()
    else:
        return {
            "dataset_id": results[0]["dataset_id"],
            "obs_stats": pd.concat([r["obs_stats"] for r in results], verify_integrity=True),
            "var_stats": reduce(
                lambda a, b: a.add(b, fill_value=0).astype(np.int64),
                (r["var_stats"] for r in results),
            ),
            "presence": list(itertools.chain(*[r["presence"] for r in results])),
        }


def reduce_X_stats_binop(a: XReduction, b: XReduction) -> XReduction:
    assert a["dataset_id"] == b["dataset_id"]
    return reduce_X_stats_chunk((a, b))


def compute_X_file_stats(
    xreduction: XReduction, n_obs: int, dataset_id: str, dataset_soma_joinid: int, eb_name: str
) -> Sequence[XReduction]:
    """Add file-stats to XReduction."""
    res = xreduction.copy()

    assert len(res["presence"]) == 0  # should only be called once per dataset
    assert res["obs_stats"].index.is_unique  # should only have one value per cell
    assert len(res["obs_stats"]) == n_obs

    obs_stats = res["obs_stats"]
    var_stats = res["var_stats"]
    obs_stats["n_measured_vars"] = (var_stats.nnz > 0).sum()
    var_stats.loc[var_stats.nnz > 0, "n_measured_obs"] = n_obs
    res["presence"].append(
        PresenceResult(
            dataset_id,
            dataset_soma_joinid,
            eb_name,
            (var_stats.nnz > 0).to_numpy(),
            var_stats.index.to_numpy(),
        ),
    )
    return (res,)


# Controls partitioning/chunking of the X array processsing:
#   REDUCE_X_MAJOR_ROW_STRIDE: the max row stride for individual tasks. Primarily affects available parallelism
#       by splitting very large datasets into multiple tasks.
#   REDUCE_X_MINOR_NNZ_STRIDE: the max nnz (value) stride used in reducing X. Drives peak memory use and TileDB
#       fragment size.
#
# An important side-effect of these parameters is the number and size of TileDB fragments created. As fragment count
# increases, consolidation time increases non-linearly. Therefore, there is a significant tradeoff between per-task
# memory use and number of fragments. These values are currently tuned for (very roughly) 128GiB/task as a maximum
# memory budget for full populated "chunks".
#
# TODO: when https://github.com/single-cell-data/TileDB-SOMA/issues/2054 is implemented, write each major stride
# as a single fragment. This would allow a much smaller minor stride, without causing fragment count to increase.
#
# See also: MEMORY_BUDGET in the `build_soma.build()` function
#
REDUCE_X_MAJOR_ROW_STRIDE: int = 2_000_000
REDUCE_X_MINOR_NNZ_STRIDE: int = 2**30


@logit(logger, msg="{2.filename}, {3}")
def dispatch_X_chunk(
    dataset_id: str,
    experiment_uri: str,
    adata: AnnDataProxy,
    row_start: int,
    n_rows: int,
    dataset_obs_joinid_start: int,
    global_var_joinids: pd.DataFrame,
) -> XReduction:
    """Read a chunk of an AnnData, pull out stats, and save as both raw & normalized layer."""
    # result accumulator
    result: XReduction = {
        "dataset_id": dataset_id,
        "obs_stats": pd.DataFrame(),
        "var_stats": pd.DataFrame(),
        "presence": [],
    }

    # Index the AnnData var coordinates into SOMA space
    local_var_joinids = adata.var.join(global_var_joinids).soma_joinid.to_numpy()
    assert (local_var_joinids >= 0).all(), f"Illegal join id, {dataset_id}"

    _is_full_gene_assay = np.isin(adata.obs.assay_ontology_term_id.to_numpy(), FULL_GENE_ASSAY)
    if _is_full_gene_assay.any():
        is_full_gene_assay: npt.NDArray[np.bool_] | None = _is_full_gene_assay
        feature_length = adata.var.feature_length.to_numpy()
    else:
        is_full_gene_assay = None
        feature_length = None

    minor_stride = clamp(int(REDUCE_X_MINOR_NNZ_STRIDE // (adata.get_estimated_density() * adata.n_vars)), 1, n_rows)
    sigma = np.finfo(np.float32).smallest_subnormal

    def getijd(
        X: sparse.csr_matrix | sparse.csc_matrix,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32 | np.int64], npt.NDArray[np.float32]]:
        X = X.tocoo()
        return X.row.astype(np.int64), X.col, X.data.astype(np.float32)

    path_to_X = (experiment_uri, "ms", MEASUREMENT_RNA_NAME, "X")
    end_idx = min(row_start + n_rows, adata.n_obs)
    n_chunks = math.ceil((end_idx - row_start) / minor_stride + 0.5)
    for idx in range(row_start, end_idx, minor_stride):
        logger.info(f"processing X {adata.filename}, {row_start}, chunk {(idx-row_start)//minor_stride} of {n_chunks}")

        # get the chunk
        adata_chunk = adata[idx : min(idx + minor_stride, end_idx)]
        n_obs, n_vars = adata_chunk.n_obs, adata_chunk.n_vars
        assert n_obs <= n_rows and n_obs <= minor_stride
        X = adata_chunk.X
        del adata_chunk

        # clean up X
        if isinstance(X, np.ndarray):
            X = sparse.csr_matrix(X)
        # force CSR - other code assumes this format (e.g., to allow indexing)
        X = X.tocsr()
        X.eliminate_zeros()  # in-place operation
        assert X.shape == (n_obs, n_vars)
        assert is_nonnegative_integral(X), "Found non-integer or negative valued data in X chunk"
        gc.collect()

        # Accumulate various statistics and summary information
        chunk_obs_joinid_start = idx + dataset_obs_joinid_start
        _obs_stats, _var_stats = _get_axis_stats(X, chunk_obs_joinid_start, local_var_joinids)
        assert len(_obs_stats) == n_obs
        assert len(_var_stats) == n_vars
        result = reduce_X_stats_chunk(
            [
                result,
                {
                    "dataset_id": dataset_id,
                    "obs_stats": _obs_stats,
                    "var_stats": _var_stats,
                    "presence": [],  # this is handled in the file reducer
                },
            ]
        )
        del _obs_stats, _var_stats

        xI, xJ, xD = getijd(X)
        assert n_obs == X.shape[0]
        del X
        gc.collect()

        if is_full_gene_assay is not None:
            assert feature_length is not None
            is_full_gene_assay_mask = is_full_gene_assay[idx : idx + minor_stride]
            xNormD = np.where(is_full_gene_assay_mask[xI], xD / feature_length[xJ], xD).astype(np.float32)
            xNormD = _divide_by_row_sum(n_obs, xI, xNormD)  # in-place operation
        else:
            xNormD = _divide_by_row_sum(n_obs, xI, xD.copy())  # in-place operation
        _roundHalfToEven(xNormD, keepbits=15)  # in-place operation
        xNormD[xNormD == 0] = sigma

        # reindex coordinates
        xI += idx + dataset_obs_joinid_start
        xJ = local_var_joinids[xJ]

        # and save to respective layer
        with soma.open(urlcat(*path_to_X, "raw"), mode="w", context=SOMA_TileDB_Context()) as X_raw:
            X_raw.write(pa.Table.from_pydict({"soma_dim_0": xI, "soma_dim_1": xJ, "soma_data": xD}))
        del xD
        gc.collect()
        with soma.open(urlcat(*path_to_X, "normalized"), mode="w", context=SOMA_TileDB_Context()) as X_normalized:
            X_normalized.write(pa.Table.from_pydict({"soma_dim_0": xI, "soma_dim_1": xJ, "soma_data": xNormD}))

        del xI, xJ, xNormD
        gc.collect()

    return result


def _reduce_X_matrices(
    base_path: str,
    datasets: list[Dataset],
    experiment_builders: list[ExperimentBuilder],
) -> dict[str, list[tuple[str, XReduction]]]:
    """Helper function for populate_X_layers."""

    def read_and_dispatch_partial_h5ad(
        dataset_id: str,
        dataset_h5ad_path: str,
        experiment_uri: str,
        row_start: int,
        n_rows: int,
        dataset_obs_joinid_start: int,
        filter_spec: AnnDataFilterSpec,
        global_var_joinids: pd.DataFrame,
    ) -> XReduction:
        return dispatch_X_chunk(
            dataset_id,
            experiment_uri,
            open_anndata(
                dataset_h5ad_path,
                filter_spec=filter_spec,
                base_path=base_path,
                var_column_names=("_index", "feature_length"),
            ),
            row_start,
            n_rows,
            dataset_obs_joinid_start,
            global_var_joinids,
        )

    per_eb_results = {}
    for eb in experiment_builders:
        if eb.n_var == 0:
            # edge case for test builds that have no data for an entire experiment (organism)
            continue

        assert eb.global_var_joinids is not None
        global_var_joinids = dask.delayed(eb.global_var_joinids)

        read_file_chunks = [
            (
                d.dataset_id,
                d.dataset_h5ad_path,
                eb.experiment_uri,
                chunk,
                REDUCE_X_MAJOR_ROW_STRIDE,
                eb.dataset_obs_joinid_start[d.dataset_id],
                eb.specification.anndata_cell_filter_spec,
            )
            for d in datasets
            if d.dataset_id in eb.dataset_obs_joinid_start
            for chunk in range(0, eb.dataset_n_obs[d.dataset_id], REDUCE_X_MAJOR_ROW_STRIDE)
        ]
        per_eb_results[eb.name] = (
            dask.bag.from_sequence(read_file_chunks)
            .starmap(read_and_dispatch_partial_h5ad, global_var_joinids=global_var_joinids)
            .foldby("dataset_id", reduce_X_stats_binop)
        )

    result: dict[str, list[tuple[str, XReduction]]]
    (result,) = dask.compute(per_eb_results)
    return result


def populate_X_layers(
    assets_path: str,
    datasets: list[Dataset],
    experiment_builders: list[ExperimentBuilder],
    args: CensusBuildArgs,
) -> None:
    """Process X layers for all datasets. Includes saving raw/normalized SOMA arrays,
    and reducing obs/var axis stats from X data.
    """
    datasets_by_id = {d.dataset_id: d for d in datasets}
    per_eb_results = _reduce_X_matrices(assets_path, datasets, experiment_builders)

    for eb in experiment_builders:
        if eb.name not in per_eb_results:
            continue

        # add per-dataset stats to each per-dataset XReduction
        eb_result: list[XReduction] = []
        for dataset_id, xreduction in per_eb_results[eb.name]:
            assert dataset_id == xreduction["dataset_id"]
            d = datasets_by_id[dataset_id]
            eb_result.extend(
                compute_X_file_stats(
                    xreduction,
                    n_obs=eb.dataset_n_obs[d.dataset_id],
                    dataset_id=d.dataset_id,
                    dataset_soma_joinid=d.soma_joinid,
                    eb_name=eb.name,
                )
            )

        eb_summary = reduce_X_stats_chunk(eb_result)
        assert isinstance(eb.obs_df, pd.DataFrame)
        eb.obs_df.update(eb_summary["obs_stats"])
        assert isinstance(eb.var_df, pd.DataFrame)
        eb.var_df.loc[
            eb_summary["var_stats"].index.to_numpy(),
            eb_summary["var_stats"].columns.to_list(),
        ] += eb_summary["var_stats"]

        for presence in eb_summary["presence"]:
            assert presence.eb_name == eb.name
            eb.presence[presence.dataset_soma_joinid] = (
                presence.data,
                presence.cols,
            )


class SummaryStats(TypedDict):
    total_cell_count: int
    unique_cell_count: int
    number_donors: dict[str, int]


def get_summary_stats(experiment_builders: Sequence[ExperimentBuilder]) -> SummaryStats:
    return {
        "total_cell_count": sum(e.n_obs for e in experiment_builders),
        "unique_cell_count": sum(e.n_unique_obs for e in experiment_builders),
        "number_donors": {e.name: e.n_donors for e in experiment_builders},
    }


def add_tissue_mapping(obs_df: pd.DataFrame) -> None:
    """Inplace addition of tissue_general-related columns."""
    # UBERON tissue term mapper
    from .tissue_mapper import TissueMapper

    tissue_mapper: TissueMapper = TissueMapper()

    tissue_ids = obs_df.tissue_ontology_term_id.unique()

    # Map specific ID -> general ID
    tissue_general_id_map = {id: tissue_mapper.get_high_level_tissue(id) for id in tissue_ids}
    assert all(tissue_general_id_map.values()), "Unable to generalize all tissue types"
    obs_df["tissue_general_ontology_term_id"] = obs_df.tissue_ontology_term_id.map(tissue_general_id_map)

    # Assign general label
    tissue_general_label_map = {
        id: tissue_mapper.get_label_from_writable_id(id) for id in tissue_general_id_map.values()
    }
    obs_df["tissue_general"] = obs_df.tissue_general_ontology_term_id.map(tissue_general_label_map)


def reopen_experiment_builders(
    experiment_builders: list[ExperimentBuilder], mode: OpenMode = "w"
) -> Generator[ExperimentBuilder, None, None]:
    """Re-opens all ExperimentBuilder's `experiment` for writing as a Generator, allowing iterating code to use
    the experiment for writing, without having to explicitly close it.
    """
    with ExitStack() as experiments_stack:
        for eb in experiment_builders:
            # open experiments for write and ensure they are closed when exiting
            assert eb.experiment is None or eb.experiment.closed
            eb.experiment = soma.Experiment.open(eb.experiment_uri, mode, context=SOMA_TileDB_Context())
            experiments_stack.enter_context(eb.experiment)

        for eb in experiment_builders:
            yield eb


@numba.jit(nopython=True, nogil=True)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _divide_by_row_sum(
    n_rows: int,
    d0: npt.NDArray[np.int64],
    data: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """IMPORTANT: in-place operation. Divide each value by the sum of the row."""
    row_sum = np.zeros((n_rows,), dtype=np.float64)
    for i in range(len(d0)):
        row_sum[d0[i]] += data[i]

    for i in range(len(d0)):
        data[i] = data[i] / row_sum[d0[i]]

    return data


@numba.jit(nopython=True, nogil=True)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _roundHalfToEven(a: npt.NDArray[np.float32], keepbits: int) -> npt.NDArray[np.float32]:
    """Generate reduced precision floating point array, with round half to even.
    IMPORANT: In-place operation.

    Ref: https://gmd.copernicus.org/articles/14/377/2021/gmd-14-377-2021.html
    """
    assert a.dtype is np.dtype(np.float32)  # code below assumes IEEE 754 float32
    nmant = 23
    bits = 32
    if keepbits < 1 or keepbits >= nmant:
        return a
    maskbits = nmant - keepbits
    full_mask = (1 << bits) - 1
    mask = (full_mask >> maskbits) << maskbits
    half_quantum1 = (1 << (maskbits - 1)) - 1

    b = a.view(np.int32)
    b += ((b >> maskbits) & 1) + half_quantum1
    b &= mask
    return a
