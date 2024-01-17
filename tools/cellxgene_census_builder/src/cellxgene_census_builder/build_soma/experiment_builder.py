import concurrent.futures
import gc
import logging
from contextlib import ExitStack
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    cast,
    overload,
)

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
from typing_extensions import Self

from ..build_state import CensusBuildArgs
from ..util import log_process_resource_status, urlcat
from .anndata import AnnDataFilterSpec, make_anndata_cell_filter, open_anndata
from .datasets import Dataset
from .globals import (
    CENSUS_OBS_PLATFORM_CONFIG,
    CENSUS_OBS_STATS_COLUMNS,
    CENSUS_OBS_TABLE_SPEC,
    CENSUS_VAR_PLATFORM_CONFIG,
    CENSUS_VAR_TABLE_SPEC,
    CENSUS_X_LAYERS,
    CENSUS_X_LAYERS_PLATFORM_CONFIG,
    CXG_OBS_COLUMNS_READ,
    CXG_VAR_COLUMNS_READ,
    DONOR_ID_IGNORE,
    FEATURE_DATASET_PRESENCE_MATRIX_NAME,
    MEASUREMENT_RNA_NAME,
    SMART_SEQ,
    SOMA_TileDB_Context,
)
from .mp import (
    ResourcePoolProcessExecutor,
    create_process_pool_executor,
    create_resource_pool_executor,
    log_on_broken_process_pool,
    n_workers_from_memory_budget,
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


AccumulateXResult = Tuple[PresenceResult, AxisStats]
AccumulateXResults = Sequence[AccumulateXResult]


def _assert_open_for_write(obj: Optional[somacore.SOMAObject]) -> None:
    assert obj is not None
    assert obj.exists(obj.uri)
    assert obj.mode == "w"
    assert not obj.closed


@attrs.define(frozen=True)
class ExperimentSpecification:
    """
    Declarative "specification" of a SOMA experiment. This is a read-only
    specification, independent of the datasets used to build the census.

    Parameters:
    * experiment "name" (eg, 'human'), must be unique in all experiments.
    * an AnnData filter used to cherry pick data for the experiment
    * external reference data used to build the experiment, e.g., gene length data

    Usage: to create, use the factory method `ExperimentSpecification.create(...)`
    """

    name: str
    anndata_cell_filter_spec: AnnDataFilterSpec

    @classmethod
    def create(
        cls,
        name: str,
        anndata_cell_filter_spec: AnnDataFilterSpec,
    ) -> Self:
        """Factory method. Do not instantiate the class directly."""
        return cls(name, anndata_cell_filter_spec)


class ExperimentBuilder:
    """
    Class that embodies the operators and state to build an Experiment.
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
        self.obs_df: Optional[pd.DataFrame] = None
        self.var_df: Optional[pd.DataFrame] = None
        self.dataset_obs_joinid_start: Dict[str, int] = {}
        self.census_summary_cell_counts: pd.DataFrame = init_summary_counts_accumulator()
        self.experiment: Optional[soma.Experiment] = None  # initialized in create()
        self.experiment_uri: Optional[str] = None  # initialized in create()
        self.global_var_joinids: Optional[pd.DataFrame] = None
        self.presence: Dict[int, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]] = {}

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

        obs_df = CENSUS_OBS_TABLE_SPEC.recategoricalize(self.obs_df)
        obs_schema = CENSUS_OBS_TABLE_SPEC.to_arrow_schema(obs_df)

        # create `obs`
        self.experiment.add_new_dataframe(
            "obs",
            schema=obs_schema,
            index_column_names=["soma_joinid"],
            platform_config=CENSUS_OBS_PLATFORM_CONFIG,
        )

        if obs_df is None or obs_df.empty:
            logger.info(f"{self.name}: empty obs dataframe")
        else:
            logger.debug(f"experiment {self.name} obs = {obs_df.shape}")
            assert not np.isnan(obs_df.nnz.to_numpy()).any()  # sanity check
            pa_table = pa.Table.from_pandas(
                obs_df, preserve_index=False, columns=list(CENSUS_OBS_TABLE_SPEC.field_names())
            )
            self.experiment.obs.write(pa_table)

    def write_var_dataframe(self) -> None:
        logger.info(f"{self.name}: writing var dataframe")
        assert self.experiment is not None
        _assert_open_for_write(self.experiment)

        rna_measurement = self.experiment.ms[MEASUREMENT_RNA_NAME]

        var_df = CENSUS_VAR_TABLE_SPEC.recategoricalize(self.var_df)
        var_schema = CENSUS_VAR_TABLE_SPEC.to_arrow_schema(var_df)

        # create `var` in the measurement
        rna_measurement.add_new_dataframe(
            "var",
            schema=var_schema,
            index_column_names=["soma_joinid"],
            platform_config=CENSUS_VAR_PLATFORM_CONFIG,
        )

        if var_df is None or var_df.empty:
            logger.info(f"{self.name}: empty var dataframe")
        else:
            logger.debug(f"experiment {self.name} var = {var_df.shape}")
            pa_table = pa.Table.from_pandas(
                var_df, preserve_index=False, columns=list(CENSUS_VAR_TABLE_SPEC.field_names())
            )
            rna_measurement.var.write(pa_table)

    def create_X_with_layers(self) -> None:
        """
        Create layers in ms['RNA']/X
        """
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

    def populate_presence_matrix(self, datasets: List[Dataset]) -> None:
        """
        Save presence matrix per Experiment
        """
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

    def write_X_normalized(self, args: CensusBuildArgs) -> None:
        assert self.experiment is not None
        if self.obs_df is None or self.n_obs == 0:
            return

        logger.info(f"Write X normalized: {self.name} - starting")
        is_smart_seq = np.isin(self.obs_df.assay_ontology_term_id.to_numpy(), SMART_SEQ)
        assert self.var_df is not None
        feature_length = self.var_df.feature_length.to_numpy()

        if args.config.multi_process:
            WRITE_NORM_STRIDE = 2**18  # controls TileDB fragment size, which impacts consolidation time
            mem_budget = (
                # (20 bytes per COO X stride X typical-nnz X overhead) + static-allocation + passed-data-size
                int(20 * WRITE_NORM_STRIDE * 4000 * 2)
                + (3 * 1024**3)
                + feature_length.nbytes
                + is_smart_seq.nbytes
            )
            n_workers = n_workers_from_memory_budget(args, mem_budget)
            with create_process_pool_executor(args, max_workers=n_workers) as pe:
                futures = [
                    pe.submit(
                        _write_X_normalized,
                        (self.experiment.uri, start_id, WRITE_NORM_STRIDE, feature_length, is_smart_seq),
                    )
                    for start_id in range(0, self.n_obs, WRITE_NORM_STRIDE)
                ]
                for n, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                    log_on_broken_process_pool(pe)
                    # prop exceptions by calling result
                    f.result()
                    logger.info(f"Write X normalized ({self.name}): {n} of {len(futures)} complete.")
                    log_process_resource_status()

        else:
            _write_X_normalized((self.experiment.uri, 0, self.n_obs, feature_length, is_smart_seq))

        logger.info(f"Write X normalized: {self.name} - finished")
        log_process_resource_status()


def accumulate_axes_dataframes(
    base_path: str,
    datasets: List[Dataset],
    experiment_builders: List[ExperimentBuilder],
) -> List[tuple[ExperimentBuilder, tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Two parallel operations.
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
            obs_column_names=CXG_OBS_COLUMNS_READ,
            var_column_names=CXG_VAR_COLUMNS_READ,
        ) as adata:
            filtered_adata = make_anndata_cell_filter(spec.anndata_cell_filter_spec)(adata)
            logging.debug(f"{dataset.dataset_id}/{spec.name} - found {filtered_adata.n_obs} cells")

            # Skip this dataset if there are not cells after filtering
            if filtered_adata.n_obs == 0:
                logger.debug(f"{spec.name} - H5AD has no data after filtering, skipping {dataset.dataset_id}")
                return pd.DataFrame(), pd.DataFrame()

            obs_df = filtered_adata.obs.copy()
            obs_df["dataset_id"] = dataset.dataset_id

            var_df = (
                filtered_adata.var.copy()
                .rename_axis("feature_id")
                .reset_index()[["feature_id", "feature_name", "feature_length"]]
            )

            return obs_df, var_df

    datasets_bag = dask.bag.from_sequence(datasets)
    df_pairs_per_eb: List[tuple[pd.DataFrame, pd.DataFrame]] = dask.compute(
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
                    pd.concat(
                        cast(List[pd.DataFrame], [df_pair[0] for df_pair in df_pairs if not df_pair[0].empty]),
                        ignore_index=True,
                        join="inner",
                    ),
                    pd.concat(
                        cast(List[pd.DataFrame], [df_pair[1] for df_pair in df_pairs if not df_pair[1].empty]),
                        ignore_index=True,
                        join="inner",
                    ).drop_duplicates(ignore_index=True),
                )
                for df_pairs in df_pairs_per_eb
            ],
        )
    )


def post_acc_axes_processing(accumulated: List[tuple[ExperimentBuilder, tuple[pd.DataFrame, pd.DataFrame]]]) -> None:
    """
    Processing steps post-accumulation of all axes dataframes. Includes:
    * assign soma_joinids
    * add derived or summary columns
    * generate summary and/or working data for the experiment_builder
    """

    def add_placeholder_columns(df: pd.DataFrame, table_spec: TableSpec, default: Any) -> None:
        for key in table_spec.field_names():
            if key not in df:
                df[key] = np.full(
                    (len(df),),
                    default,
                    dtype=table_spec.field(key).to_pandas_dtype(ignore_dict_type=True),
                )

    def per_dataset_summary_counts(eb: ExperimentBuilder, obs: pd.DataFrame) -> None:
        for _, obs_slice in obs.groupby("dataset_id"):
            assert obs_slice.soma_joinid.max() - obs_slice.soma_joinid.min() + 1 == len(obs_slice)
            eb.census_summary_cell_counts = accumulate_summary_counts(eb.census_summary_cell_counts, obs_slice)

    for eb, (obs, var) in accumulated:
        obs["soma_joinid"] = range(0, len(obs))
        var["soma_joinid"] = range(0, len(var))

        add_tissue_mapping(obs)  # add tissue mapping (e.g., tissue_general term)

        # add columns to be completed later, e.g., summary stats such as mean of X
        add_placeholder_columns(obs, CENSUS_OBS_TABLE_SPEC, default=np.nan)
        add_placeholder_columns(var, CENSUS_VAR_TABLE_SPEC, default=0)

        # compute intermediate values used later in the build
        eb.n_datasets = obs.dataset_id.nunique()
        eb.n_unique_obs = (obs.is_primary_data == True).sum()  # noqa: E712
        eb.n_donors = obs[~obs.donor_id.isin(DONOR_ID_IGNORE)].groupby("dataset_id").donor_id.nunique().sum()
        eb.dataset_obs_joinid_start = obs.groupby("dataset_id").soma_joinid.min().to_dict()
        eb.global_var_joinids = var[["feature_id", "soma_joinid"]].set_index("feature_id")

        # gather per-dataset summary statistics e.g., cell type counts, etc.
        per_dataset_summary_counts(eb, obs)

        eb.obs_df = obs
        eb.var_df = var
        eb.n_obs = len(obs)
        eb.n_var = len(var)


def _get_axis_stats(
    raw_X: Union[sparse.spmatrix, npt.NDArray[np.float32]],
    dataset_obs_joinid_start: int,
    local_var_joinids: npt.NDArray[np.int64],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate obs and var summary stats, e.g., raw_sum, etc.

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


@attrs.define(frozen=True)
class AccumXEBParams:
    """The ExperimentBuilder params/info we want to pass to our worker
    processes. Do not just send an ExperimentBuilder as it is a huge
    object that takes forever to pickle.
    """

    name: str
    n_obs: int
    n_var: int
    anndata_cell_filter_spec: AnnDataFilterSpec
    global_var_joinids: Optional[pd.DataFrame]
    experiment_uri: str


# Read dim0 coords stride
ACCUM_X_STRIDE = 125_000


def _accumulate_all_X_layers(
    assets_path: str,
    dataset: Dataset,
    experiment_builders: List[AccumXEBParams],
    dataset_obs_joinid_starts: List[Union[None, int]],
    ms_name: str,
    progress: Tuple[int, int],
) -> AccumulateXResults:
    """
    For this dataset, save all X layer information for each Experiment. This currently
    includes:
        X['raw'] - raw counts

    Also accumulates presence information per dataset.

    This is a helper function for ExperimentBuilder.accumulate_X
    """
    logger.info(f"Saving X layer for dataset - start {dataset.dataset_id} ({progress[0]} of {progress[1]})")
    unfiltered_ad = open_anndata(
        dataset, base_path=assets_path, include_filter_columns=True, var_column_names=("_index",)
    )

    results: List[AccumulateXResult] = []
    for eb, dataset_obs_joinid_start in zip(experiment_builders, dataset_obs_joinid_starts):
        if dataset_obs_joinid_start is None:
            # this dataset has no data for this experiment
            continue

        if eb.n_var == 0:
            # edge case for test builds that have no data for an entire experiment (organism)
            continue

        assert eb.global_var_joinids is not None

        anndata_cell_filter = make_anndata_cell_filter(eb.anndata_cell_filter_spec)
        ad = anndata_cell_filter(unfiltered_ad)
        if ad.n_obs == 0:
            continue

        # save X['raw']
        layer_name = "raw"
        logger.info(
            f"{eb.name}: saving X layer '{layer_name}' for dataset '{dataset.dataset_id}' "
            f"({progress[0]} of {progress[1]})"
        )
        local_var_joinids = ad.var.join(eb.global_var_joinids).soma_joinid.to_numpy()
        assert (local_var_joinids >= 0).all(), f"Illegal join id, {dataset.dataset_id}"

        # accumulators
        obs_stats: pd.DataFrame = pd.DataFrame()
        var_stats: pd.DataFrame = pd.DataFrame()

        for idx in range(0, ad.n_obs, ACCUM_X_STRIDE):
            logger.debug(f"{eb.name}/{layer_name}: X chunk {idx//ACCUM_X_STRIDE + 1} {dataset.dataset_id}")

            X = ad[idx : idx + ACCUM_X_STRIDE].X
            if isinstance(X, np.ndarray):
                X = sparse.csr_matrix(X)

            assert is_nonnegative_integral(X), f"{dataset.dataset_id} contains non-integer or negative valued data"

            X.eliminate_zeros()
            gc.collect()

            # accumulate obs/var axis stats
            _obs_stats, _var_stats = _get_axis_stats(X, idx + dataset_obs_joinid_start, local_var_joinids)
            obs_stats = pd.concat([obs_stats, _obs_stats])
            var_stats = var_stats.add(_var_stats, fill_value=0).astype(np.int64)
            del _obs_stats, _var_stats
            gc.collect()

            # remap to match axes joinids
            X = X.tocoo()
            row = X.row.astype(np.int64) + idx + dataset_obs_joinid_start
            assert (row >= 0).all()
            col = local_var_joinids[X.col]
            assert (col >= 0).all()
            data = X.data
            del X
            gc.collect()

            with soma.Experiment.open(eb.experiment_uri, "w", context=SOMA_TileDB_Context()) as experiment:
                experiment.ms[ms_name].X[layer_name].write(
                    pa.Table.from_pydict(
                        {
                            "soma_dim_0": row,
                            "soma_dim_1": col,
                            "soma_data": data,
                        }
                    )
                )

            del row, col, data
            gc.collect()

        # Save presence information by dataset_id
        assert dataset.soma_joinid >= 0  # i.e., it was assigned prior to this step
        assert ad.n_vars <= eb.n_var
        obs_stats["n_measured_vars"] = (var_stats.nnz > 0).sum()
        var_stats["n_measured_obs"] = np.zeros(
            (len(var_stats),), dtype=CENSUS_VAR_TABLE_SPEC.field("n_measured_obs").to_pandas_dtype()
        )
        var_stats.n_measured_obs[var_stats.nnz > 0] = ad.n_obs

        # sanity check on stats
        assert len(var_stats) == ad.n_vars
        assert len(obs_stats) == ad.n_obs
        assert (
            var_stats.n_measured_obs[var_stats.n_measured_obs != 0] == ad.n_obs
        ).all(), f"n_measured_obs mismatch: {eb.name}:{dataset.dataset_id}"
        assert (
            obs_stats.n_measured_vars == np.count_nonzero(var_stats.nnz > 0)
        ).all(), f"n_measured_vars mismatch: {eb.name}:{dataset.dataset_id}"

        results.append(
            (
                PresenceResult(
                    dataset.dataset_id,
                    dataset.soma_joinid,
                    eb.name,
                    (var_stats.nnz > 0).to_numpy(),
                    var_stats.index.to_numpy(),
                ),
                AxisStats(eb.name, obs_stats, var_stats),
            )
        )

        # tidy
        del ad, local_var_joinids, obs_stats, var_stats
        gc.collect()

    logger.debug(f"Saving X layer for dataset - finish {dataset.dataset_id} ({progress[0]} of {progress[1]})")
    log_process_resource_status()
    return results


@overload
def _accumulate_X(
    assets_path: str,
    dataset: Dataset,
    experiment_builders: List["ExperimentBuilder"],
    progress: Tuple[int, int],
) -> AccumulateXResults:
    ...


@overload
def _accumulate_X(
    assets_path: str,
    dataset: Dataset,
    experiment_builders: List["ExperimentBuilder"],
    progress: Tuple[int, int],
    executor: Optional[ResourcePoolProcessExecutor],
) -> concurrent.futures.Future[AccumulateXResults]:
    ...


def _accumulate_X(
    assets_path: str,
    dataset: Dataset,
    experiment_builders: List["ExperimentBuilder"],
    progress: Tuple[int, int],
    executor: Optional[ResourcePoolProcessExecutor] = None,
) -> Union[concurrent.futures.Future[AccumulateXResults], AccumulateXResults]:
    """
    Save X layer data for a single AnnData, for all Experiments. Return a future if
    executor is specified, otherwise immediately do the work.
    """

    # build params to pass to child workers - this avoids pickling unecessary
    # data (or data that can't be pickled)
    eb_params = []
    for eb in experiment_builders:
        # sanity checks
        assert eb.dataset_obs_joinid_start is not None
        assert eb.n_var == 0 or eb.global_var_joinids is not None
        assert eb.experiment_uri is not None
        eb_params.append(
            AccumXEBParams(
                name=eb.name,
                n_obs=eb.n_obs,
                n_var=eb.n_var,
                anndata_cell_filter_spec=eb.anndata_cell_filter_spec,
                global_var_joinids=eb.global_var_joinids,
                experiment_uri=eb.experiment_uri,
            )
        )

    dataset_obs_joinid_starts = [
        eb.dataset_obs_joinid_start.get(dataset.dataset_id, None) for eb in experiment_builders
    ]

    if executor is not None:
        return executor.submit(
            # memory budget:
            #   X slices: stride * avg_var_nnz * 20 bytes * overhead
            # + anndata obs/var:  (n_var * 5 cols * 8 + n_obs * 4 * 8) * overhead
            # + stats space: 5 col * n_obs * 8 bytes * overhead
            # + working space: fixed value
            int(
                (
                    max(dataset.mean_genes_per_cell, 3000)
                    * min(ACCUM_X_STRIDE, dataset.dataset_total_cell_count)
                    * 20
                    * 8
                )
                + (100_000 * 40 + dataset.dataset_total_cell_count * 32) * 8
                + (dataset.dataset_total_cell_count * 40) * 8
                + (2 * 1024**3)
            ),
            _accumulate_all_X_layers,
            assets_path,
            dataset,
            eb_params,
            dataset_obs_joinid_starts,
            MEASUREMENT_RNA_NAME,
            progress,
        )
    else:
        return _accumulate_all_X_layers(
            assets_path,
            dataset,
            eb_params,
            dataset_obs_joinid_starts,
            MEASUREMENT_RNA_NAME,
            progress,
        )


def populate_X_layers(
    assets_path: str,
    datasets: List[Dataset],
    experiment_builders: List[ExperimentBuilder],
    args: CensusBuildArgs,
) -> None:
    """
    Do all X layer processing for all Experiments. Also accumulate presence matrix data for later writing.
    """
    # populate X layers
    logger.debug("populate_X_layers begin")
    results: List[AccumulateXResult] = []
    if args.config.multi_process:
        # reserve memory to accumulate the stats
        n_obs = sum(eb.n_obs for eb in experiment_builders)
        n_stats_per_obs = len(CENSUS_OBS_STATS_COLUMNS) * 8  # all 64 bit stats
        total_memory_budget = args.config.memory_budget - (n_obs * n_stats_per_obs * 2)

        with create_resource_pool_executor(args, max_resources=total_memory_budget) as pe:
            futures = {
                _accumulate_X(
                    assets_path,
                    dataset,
                    experiment_builders,
                    progress=(n, len(datasets)),
                    executor=pe,
                ): dataset
                for n, dataset in enumerate(datasets, start=1)
            }

            for n, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                log_on_broken_process_pool(pe)
                # propagate exceptions - not expecting any other return values
                results += f.result()
                logger.info(f"populate X for dataset {futures[f].dataset_id} ({n} of {len(futures)}) complete.")
                log_process_resource_status()

    else:
        for n, dataset in enumerate(datasets, start=1):
            results += _accumulate_X(assets_path, dataset, experiment_builders, progress=(n, len(datasets)))

    eb_by_name = {e.name: e for e in experiment_builders}

    # sanity check
    for eb in experiment_builders:
        assert eb.obs_df is None or np.array_equal(eb.obs_df.index.to_numpy(), eb.obs_df.soma_joinid.to_numpy())

    logger.debug("populate_X_layers - begin presence summary")
    for presence, _ in results:
        eb_by_name[presence.eb_name].presence[presence.dataset_soma_joinid] = (
            presence.data,
            presence.cols,
        )

    logger.debug("populate_X_layers - begin axis stats summary")
    for _, axis_stats in results:
        eb = eb_by_name[axis_stats.eb_name]
        if eb.obs_df is not None:
            eb.obs_df.update(axis_stats.obs_stats)
        if eb.var_df is not None:
            eb.var_df.loc[
                axis_stats.var_stats.index.to_numpy(),
                axis_stats.var_stats.columns.to_list(),
            ] += axis_stats.var_stats

    logger.debug("populate_X_layers finish")
    log_process_resource_status()


class SummaryStats(TypedDict):
    total_cell_count: int
    unique_cell_count: int
    number_donors: Dict[str, int]


def get_summary_stats(experiment_builders: Sequence[ExperimentBuilder]) -> SummaryStats:
    return {
        "total_cell_count": sum(e.n_obs for e in experiment_builders),
        "unique_cell_count": sum(e.n_unique_obs for e in experiment_builders),
        "number_donors": {e.name: e.n_donors for e in experiment_builders},
    }


def add_tissue_mapping(obs_df: pd.DataFrame) -> None:
    """Inplace addition of tissue_general-related columns"""

    # UBERON tissue term mapper
    from .tissue_mapper import TissueMapper  # type: ignore

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
    experiment_builders: List[ExperimentBuilder], mode: OpenMode = "w"
) -> Generator[ExperimentBuilder, None, None]:
    """
    Re-opens all ExperimentBuilder's `experiment` for writing as a Generator, allowing iterating code to use
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


def _write_X_normalized(args: Tuple[str, int, int, npt.NDArray[np.int64], npt.NDArray[np.bool_]]) -> None:
    """
    Helper for ExperimentBuilder.write_X_normalized.

    Read indicated rows from X['raw'], write to X['normalized']
    """
    experiment_uri, obs_joinid_start, n, feature_length, is_smart_seq = args
    logger.info(f"Write X normalized - starting block {obs_joinid_start}")

    sigma = np.finfo(np.float32).smallest_subnormal

    with soma.open(
        urlcat(experiment_uri, "ms", MEASUREMENT_RNA_NAME, "X", "raw"), mode="r", context=SOMA_TileDB_Context()
    ) as X_raw:
        with soma.open(
            urlcat(experiment_uri, "ms", MEASUREMENT_RNA_NAME, "X", "normalized"),
            mode="w",
            context=SOMA_TileDB_Context(),
        ) as X_normalized:
            for tbl, (obs_indices, _) in (
                X_raw.read(coords=(slice(obs_joinid_start, obs_joinid_start + n - 1),))
                .blockwise(axis=0, size=2**16, reindex_disable_on_axis=[1], eager=False)
                .tables()
            ):
                d0: npt.NDArray[np.int64] = tbl["soma_dim_0"].to_numpy()
                d1: npt.NDArray[np.int64] = tbl["soma_dim_1"].to_numpy()
                data: npt.NDArray[np.float32] = tbl["soma_data"].to_numpy()
                d0_index = obs_indices.to_numpy()
                del tbl, obs_indices
                data = _normalize(d0, d1, data, is_smart_seq[d0_index], feature_length)
                data = _roundHalfToEven(data, keepbits=15)
                data[data == 0] = sigma
                gc.collect()
                X_normalized.write(
                    pa.Table.from_pydict(
                        {
                            "soma_dim_0": d0_index[d0],
                            "soma_dim_1": d1,
                            "soma_data": data,
                        }
                    )
                )
                del d0_index, d0, d1, data
                gc.collect()

    logger.info(f"Write X normalized - finished block {obs_joinid_start}")


@numba.jit(nopython=True, nogil=True)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _normalize(
    d0: npt.NDArray[np.int64],
    d1: npt.NDArray[np.int64],
    data: npt.NDArray[np.float32],
    is_smart_seq: npt.NDArray[np.bool_],
    feature_length: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    # normalize and sum COO data along rows (assertion: will have full rows (along axis 0))
    norm_data = np.where(is_smart_seq[d0], data / feature_length[d1], data)
    row_sum = np.zeros((d0.max() + 1,), dtype=np.float64)
    for i in range(len(d0)):
        row_sum[d0[i]] += norm_data[i]

    result = np.empty_like(data)
    for i in range(len(d0)):
        result[i] = norm_data[i] / row_sum[d0[i]]

    return result


@numba.jit(nopython=True, nogil=True)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _roundHalfToEven(a: npt.NDArray[np.float32], keepbits: int) -> npt.NDArray[np.float32]:
    """
    Generate reduced precision floating point array, with round half to even.
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
