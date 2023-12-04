import concurrent.futures
import gc
import logging
from contextlib import ExitStack
from typing import (
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    overload,
)

import anndata
import attrs
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
    CENSUS_OBS_TERM_COLUMNS,
    CENSUS_VAR_PLATFORM_CONFIG,
    CENSUS_VAR_TERM_COLUMNS,
    CENSUS_X_LAYER_NORMALIZED_FLOAT_SCALE_FACTOR,
    CENSUS_X_LAYERS,
    CENSUS_X_LAYERS_PLATFORM_CONFIG,
    CXG_OBS_TERM_COLUMNS,
    DONOR_ID_IGNORE,
    FEATURE_DATASET_PRESENCE_MATRIX_NAME,
    MEASUREMENT_RNA_NAME,
    SOMA_TileDB_Context,
)
from .mp import (
    EagerIterator,
    ResourcePoolProcessExecutor,
    create_process_pool_executor,
    create_resource_pool_executor,
    create_thread_pool_executor,
    log_on_broken_process_pool,
    n_workers_from_memory_budget,
)
from .stats import get_obs_stats, get_var_stats
from .summary_cell_counts import (
    accumulate_summary_counts,
    init_summary_counts_accumulator,
)
from .util import array_chunker, is_nonnegative_integral


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
        self.obs_df_accumulation: List[pd.DataFrame] = []
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

        logging.info(f"{self.name}: create experiment at {urlcat(census_data.uri, self.name)}")

        self.experiment = census_data.add_new_collection(self.name, soma.Experiment)
        self.experiment_uri = self.experiment.uri

        # create `ms`
        ms = self.experiment.add_new_collection("ms")

        # create `obs`
        obs_schema = pa.schema(list(CENSUS_OBS_TERM_COLUMNS.items()))
        self.experiment.add_new_dataframe(
            "obs",
            schema=obs_schema,
            index_column_names=["soma_joinid"],
            platform_config=CENSUS_OBS_PLATFORM_CONFIG,
        )

        # make measurement and add to ms collection
        rna_measurement = ms.add_new_collection(MEASUREMENT_RNA_NAME, soma.Measurement)

        # create `var` in the measurement
        var_schema = pa.schema(list(CENSUS_VAR_TERM_COLUMNS.items()))
        rna_measurement.add_new_dataframe(
            "var",
            schema=var_schema,
            index_column_names=["soma_joinid"],
            platform_config=CENSUS_VAR_PLATFORM_CONFIG,
        )

    def filter_anndata_cells(self, ad: anndata.AnnData) -> Union[None, anndata.AnnData]:
        anndata_cell_filter = make_anndata_cell_filter(self.anndata_cell_filter_spec)
        return anndata_cell_filter(ad, need_X=False)

    def accumulate_axes(self, dataset: Dataset, ad: anndata.AnnData) -> int:
        """
        Build (accumulate) in-memory obs and var.

        Returns: number of cells that make it past the experiment filter.
        """

        assert len(ad.obs) > 0

        # Narrow columns just to minimize memory footprint. Summary cell counting
        # requires 'organism', do be careful not to delete that.
        obs_df = ad.obs[list(CXG_OBS_TERM_COLUMNS) + ["organism"]].reset_index(drop=True).copy()

        obs_df["soma_joinid"] = range(self.n_obs, self.n_obs + len(obs_df))
        obs_df["dataset_id"] = dataset.dataset_id

        # high-level tissue mapping
        add_tissue_mapping(obs_df, dataset.dataset_id)

        # add any other computed columns
        for key in CENSUS_OBS_TERM_COLUMNS:
            if key not in obs_df:
                obs_df[key] = np.full(
                    (len(obs_df),),
                    np.nan,
                    dtype=CENSUS_OBS_TERM_COLUMNS[key].to_pandas_dtype(),
                )

        # Accumulate aggregation counts
        self.census_summary_cell_counts = accumulate_summary_counts(self.census_summary_cell_counts, obs_df)

        # drop columns we don't want to write (e.g., organism)
        obs_df = obs_df[list(CENSUS_OBS_TERM_COLUMNS)]

        # accumulate obs
        self.obs_df_accumulation.append(obs_df)

        self.dataset_obs_joinid_start[dataset.dataset_id] = self.n_obs

        # Accumulate the union of all var ids/names (for raw and processed), to be later persisted.
        # NOTE: assumes raw.var is None, OR has same index as var. Currently enforced in open_anndata(),
        # but may need to evolve this logic if that assumption is not scalable.
        tv = ad.var.rename_axis("feature_id").reset_index()[["feature_id", "feature_name", "feature_length"]]
        for key in CENSUS_VAR_TERM_COLUMNS:
            if key not in tv:
                tv[key] = np.full((len(tv),), 0, dtype=CENSUS_VAR_TERM_COLUMNS[key].to_pandas_dtype())
        self.var_df = (
            pd.concat([self.var_df, tv], ignore_index=True).drop_duplicates() if self.var_df is not None else tv
        )

        self.n_obs += len(obs_df)
        self.n_unique_obs += (obs_df.is_primary_data == True).sum()  # noqa: E712

        donors = obs_df.donor_id.unique()
        self.n_donors += len(donors) - np.isin(donors, DONOR_ID_IGNORE).sum()

        self.n_datasets += 1
        return len(obs_df)

    def finalize_obs_axes(self) -> None:
        if not self.obs_df_accumulation:
            return
        self.obs_df = pd.concat(self.obs_df_accumulation, ignore_index=True)
        self.obs_df_accumulation.clear()
        assert self.n_obs == len(self.obs_df)
        gc.collect()

    def write_obs_dataframe(self) -> None:
        logging.info(f"{self.name}: writing obs dataframe")
        _assert_open_for_write(self.experiment)

        if self.obs_df is None or len(self.obs_df) == 0:
            logging.info(f"{self.name}: empty obs dataframe")
        else:
            logging.debug(f"experiment {self.name} obs = {self.obs_df.shape}")
            assert not np.isnan(self.obs_df.nnz.to_numpy()).any()  # sanity check
            pa_table = pa.Table.from_pandas(
                self.obs_df,
                preserve_index=False,
                columns=list(CENSUS_OBS_TERM_COLUMNS),
            )
            self.experiment.obs.write(pa_table)  # type:ignore

    def write_var_dataframe(self) -> None:
        logging.info(f"{self.name}: writing var dataframe")
        _assert_open_for_write(self.experiment)

        if self.var_df is None or len(self.var_df) == 0:
            logging.info(f"{self.name}: empty var dataframe")
        else:
            logging.debug(f"experiment {self.name} var = {self.var_df.shape}")
            pa_table = pa.Table.from_pandas(
                self.var_df,
                preserve_index=False,
                columns=list(CENSUS_VAR_TERM_COLUMNS),
            )
            self.experiment.ms["RNA"].var.write(pa_table)  # type:ignore

    def populate_var_axis(self) -> None:
        logging.info(f"{self.name}: populate var axis")

        # it is possible there is nothing to write
        if self.var_df is not None and len(self.var_df) > 0:
            self.var_df["soma_joinid"] = range(len(self.var_df))
            self.var_df = self.var_df.set_index("soma_joinid", drop=False)

            self.global_var_joinids = self.var_df[["feature_id", "soma_joinid"]].set_index("feature_id")
            self.n_var = len(self.var_df)
        else:
            self.n_var = 0

    def create_X_with_layers(self) -> None:
        """
        Create layers in ms['RNA']/X
        """
        logging.info(f"{self.name}: create X layers")

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
        logging.info(f"Save presence matrix for {self.name} - start")

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

            fdpm = self.experiment.ms["RNA"].add_new_sparse_ndarray(  # type:ignore
                FEATURE_DATASET_PRESENCE_MATRIX_NAME,
                type=pa.bool_(),
                shape=(max_dataset_joinid + 1, self.n_var),
            )
            fdpm.write(pa.SparseCOOTensor.from_scipy(pm))

        logging.info(f"Save presence matrix for {self.name} - finish")
        log_process_resource_status()

    def write_X_normalized(self, args: CensusBuildArgs) -> None:
        assert self.experiment is not None
        if self.obs_df is None or self.n_obs == 0:
            return

        logging.info(f"Write X normalized: {self.name} - starting")
        # grab the previously calculated sum of the X['raw'] layer
        raw_sum = self.obs_df.raw_sum.to_numpy()

        if args.config.multi_process:
            STRIDE = 1_000_000  # controls TileDB fragment size, which impacts consolidation time
            # memory budget: 3 attribute buffers * 3 threads * 100% buffer
            mem_budget = 3 * 3 * 2 * int(SOMA_TileDB_Context().tiledb_ctx.config()["soma.init_buffer_bytes"])
            n_workers = n_workers_from_memory_budget(args, mem_budget)
            with create_process_pool_executor(args, max_workers=n_workers) as pe:
                futures = [
                    pe.submit(_write_X_normalized, (self.experiment.uri, start_id, STRIDE, raw_sum))
                    for start_id in range(0, self.n_obs, STRIDE)
                ]
                for n, f in enumerate(concurrent.futures.as_completed(futures), start=1):
                    log_on_broken_process_pool(pe)
                    # prop exceptions by calling result
                    f.result()
                    logging.info(f"Write X normalized ({self.name}): {n} of {len(futures)} complete.")
                    log_process_resource_status()

        else:
            _write_X_normalized((self.experiment.uri, 0, self.n_obs, raw_sum))

        logging.info(f"Write X normalized: {self.name} - finished")
        log_process_resource_status()


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
    logging.info(f"Saving X layer for dataset - start {dataset.dataset_id} ({progress[0]} of {progress[1]})")
    gc.collect()
    _, unfiltered_ad = next(open_anndata(assets_path, [dataset], need_X=True))
    assert unfiltered_ad.isbacked is False

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

        # follow CELLxGENE 3.0 schema conventions for raw/X aliasing when only raw counts exist
        raw_X, raw_var = (ad.X, ad.var) if ad.raw is None else (ad.raw.X, ad.raw.var)

        if not is_nonnegative_integral(raw_X):
            logging.error(f"{dataset.dataset_id} contains non-integer or negative valued data")

        if isinstance(raw_X, np.ndarray):
            raw_X = sparse.csr_matrix(raw_X)

        raw_X.eliminate_zeros()

        # save X['raw']
        layer_name = "raw"
        logging.info(
            f"{eb.name}: saving X layer '{layer_name}' for dataset '{dataset.dataset_id}' "
            f"({progress[0]} of {progress[1]})"
        )
        local_var_joinids = raw_var.join(eb.global_var_joinids).soma_joinid.to_numpy()
        assert (local_var_joinids >= 0).all(), f"Illegal join id, {dataset.dataset_id}"

        for n, X in enumerate(array_chunker(raw_X), start=1):
            logging.debug(f"{eb.name}/{layer_name}: X chunk {n} {dataset.dataset_id}")
            # remap to match axes joinids
            row = X.row.astype(np.int64) + dataset_obs_joinid_start
            assert (row >= 0).all()
            col = local_var_joinids[X.col]
            assert (col >= 0).all()
            X_remap = sparse.coo_matrix((X.data, (row, col)), shape=(eb.n_obs, eb.n_var))
            with soma.Experiment.open(eb.experiment_uri, "w", context=SOMA_TileDB_Context()) as experiment:
                experiment.ms[ms_name].X[layer_name].write(pa.SparseCOOTensor.from_scipy(X_remap))
            gc.collect()

        # Save presence information by dataset_id
        assert dataset.soma_joinid >= 0  # i.e., it was assigned prior to this step
        pres_data: npt.NDArray[np.bool_] = raw_X.sum(axis=0) > 0
        if isinstance(pres_data, np.matrix):
            pres_data = pres_data.A
        pres_data = pres_data[0]
        pres_cols: npt.NDArray[np.int64] = local_var_joinids[np.arange(ad.n_vars, dtype=np.int64)]

        assert pres_data.dtype == bool
        assert pres_cols.dtype == np.int64
        assert pres_data.shape == (ad.n_vars,)
        assert pres_data.shape == pres_cols.shape
        assert ad.n_vars <= eb.n_var

        obs_stats, var_stats = _get_axis_stats(raw_X, dataset_obs_joinid_start, local_var_joinids)

        results.append(
            (
                PresenceResult(
                    dataset.dataset_id,
                    dataset.soma_joinid,
                    eb.name,
                    pres_data,
                    pres_cols,
                ),
                AxisStats(eb.name, obs_stats, var_stats),
            )
        )

        # tidy
        del ad, raw_X, raw_var, local_var_joinids, row, col, X_remap, pres_data, pres_cols, obs_stats, var_stats

    gc.collect()
    logging.debug(f"Saving X layer for dataset - finish {dataset.dataset_id} ({progress[0]} of {progress[1]})")
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
            12 * dataset.asset_h5ad_filesize,  # Heuristic value based upon empirical testing.
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
    logging.debug("populate_X_layers begin")
    results: List[AccumulateXResult] = []
    if args.config.multi_process:
        with create_resource_pool_executor(args) as pe:
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
                logging.info(f"populate X for dataset {futures[f].dataset_id} ({n} of {len(futures)}) complete.")
                log_process_resource_status()

    else:
        for n, dataset in enumerate(datasets, start=1):
            results += _accumulate_X(assets_path, dataset, experiment_builders, progress=(n, len(datasets)))

    eb_by_name = {e.name: e for e in experiment_builders}

    # sanity check
    for eb in experiment_builders:
        assert eb.obs_df is None or np.array_equal(eb.obs_df.index.to_numpy(), eb.obs_df.soma_joinid.to_numpy())

    logging.debug("populate_X_layers - begin presence summary")
    for presence, _ in results:
        eb_by_name[presence.eb_name].presence[presence.dataset_soma_joinid] = (
            presence.data,
            presence.cols,
        )

    logging.debug("populate_X_layers - begin axis stats summary")
    for _, axis_stats in results:
        eb = eb_by_name[axis_stats.eb_name]
        if eb.obs_df is not None:
            eb.obs_df.update(axis_stats.obs_stats)
        if eb.var_df is not None:
            eb.var_df.loc[
                axis_stats.var_stats.index.to_numpy(),
                axis_stats.var_stats.columns.to_list(),
            ] += axis_stats.var_stats

    logging.debug("populate_X_layers finish")
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


def add_tissue_mapping(obs_df: pd.DataFrame, dataset_id: str) -> None:
    """Inplace addition of tissue_general-related columns"""

    # UBERON tissue term mapper
    from .tissue_mapper import TissueMapper  # type: ignore

    tissue_mapper: TissueMapper = TissueMapper()

    tissue_ids = obs_df.tissue_ontology_term_id.unique()

    # Map specific ID -> general ID
    tissue_general_id_map = {id: tissue_mapper.get_high_level_tissue(id) for id in tissue_ids}
    if not all(tissue_general_id_map.values()):
        logging.error(f"{dataset_id} contains tissue types which could not be generalized.")
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


def _write_X_normalized(args: Tuple[str, int, int, npt.NDArray[np.float32]]) -> None:
    """
    Helper for ExperimentBuilder.write_X_normalized.

    Read indicated rows from X['raw'], write to X['normalized']
    """
    experiment_uri, obs_joinid_start, n, raw_sum = args
    logging.info(f"Write X normalized - starting block {obs_joinid_start}")

    """
    Adjust normlized layer to never encode zero-valued cells where the raw count
    value is greater than zero. In our current schema configuration, FloatScaleFilter
    reduces the precision of each value, storing ``round((raw_float - offset) / factor)``
    as a four byte int.

    To ensure non-zero raw values, which would _normally_ scale to zero under
    these conditions, we add the smallest possible sigma to each value (note that
    zero valued coordinates are not stored, as this is a sparse array).

    Reducing the above transformation, and assuming float32 values, the smallest sigma is
    1/2 of the scale factor (bits of precision). Accounting for IEEE float precision,
    this reduces to:
    """
    sigma = 0.5 * (CENSUS_X_LAYER_NORMALIZED_FLOAT_SCALE_FACTOR + np.finfo(np.float32).epsneg)

    with soma.open(
        urlcat(experiment_uri, "ms", MEASUREMENT_RNA_NAME, "X", "raw"), mode="r", context=SOMA_TileDB_Context()
    ) as X_raw:
        with soma.open(
            urlcat(experiment_uri, "ms", MEASUREMENT_RNA_NAME, "X", "normalized"),
            mode="w",
            context=SOMA_TileDB_Context(),
        ) as X_normalized:
            with create_thread_pool_executor(max_workers=8) as pool:
                lazy_reader = EagerIterator(
                    X_raw.read(coords=(slice(obs_joinid_start, obs_joinid_start + n - 1),)).tables(),
                    pool=pool,
                )
                lazy_divider = EagerIterator(
                    (
                        (
                            X_tbl["soma_dim_0"],
                            X_tbl["soma_dim_1"],
                            X_tbl["soma_data"].to_numpy() / raw_sum[X_tbl["soma_dim_0"]] + sigma,
                        )
                        for X_tbl in lazy_reader
                    ),
                    pool=pool,
                )
                for soma_dim_0, soma_dim_1, soma_data in lazy_divider:
                    assert np.all(soma_data > 0.0), "Found unexpected zero value in raw layer data"
                    X_normalized.write(
                        pa.Table.from_arrays(
                            [soma_dim_0, soma_dim_1, soma_data],
                            names=["soma_dim_0", "soma_dim_1", "soma_data"],
                        )
                    )

    logging.info(f"Write X normalized - finished block {obs_joinid_start}")
