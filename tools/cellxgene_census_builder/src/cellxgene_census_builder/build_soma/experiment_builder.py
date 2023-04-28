import concurrent.futures
import gc
import io
import logging
from contextlib import ExitStack
from typing import Dict, Generator, List, Optional, Sequence, Tuple, TypedDict, Union, overload

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
from ..util import urlcat
from .anndata import AnnDataFilterSpec, make_anndata_cell_filter, open_anndata
from .datasets import Dataset
from .globals import (
    CENSUS_OBS_PLATFORM_CONFIG,
    CENSUS_OBS_TERM_COLUMNS,
    CENSUS_VAR_PLATFORM_CONFIG,
    CENSUS_VAR_TERM_COLUMNS,
    CENSUS_X_LAYERS,
    CENSUS_X_LAYERS_PLATFORM_CONFIG,
    CXG_OBS_TERM_COLUMNS,
    DONOR_ID_IGNORE,
    FEATURE_DATASET_PRESENCE_MATRIX_NAME,
    MEASUREMENT_RNA_NAME,
    SOMA_TileDB_Context,
)
from .mp import create_process_pool_executor, log_on_broken_process_pool
from .source_assets import cat_file
from .summary_cell_counts import accumulate_summary_counts, init_summary_counts_accumulator
from .tissue_mapper import TissueMapper  # type: ignore
from .util import (
    anndata_ordered_bool_issue_853_workaround,
    array_chunker,
    is_nonnegative_integral,
)

# Contents:
#   dataset_id
#   dataset_soma_joinid - used as the presence row index
#   eb_name
#   data - presence COO data
#   cols - presence COO col
#
# TODO: convert this to a dataclass or namedtuple.
#
PresenceResult = Tuple[str, int, str, npt.NDArray[np.bool_], npt.NDArray[np.int64]]
PresenceResults = Tuple[PresenceResult, ...]

# UBERON tissue term mapper
tissue_mapper: TissueMapper = TissueMapper()


def _assert_open_for_write(obj: somacore.SOMAObject) -> None:
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
    gene_feature_length_uris: List[str]
    gene_feature_length: pd.DataFrame

    @classmethod
    def create(
        cls, name: str, anndata_cell_filter_spec: AnnDataFilterSpec, gene_feature_length_uris: List[str]
    ) -> Self:
        """Factory method. Do not instantiate the class directly."""
        gene_feature_length = cls._load_gene_feature_length(gene_feature_length_uris)
        logging.info(f"Loaded gene lengths external reference for {name}, {len(gene_feature_length)} genes.")
        return cls(name, anndata_cell_filter_spec, gene_feature_length_uris, gene_feature_length)

    @classmethod
    def _load_gene_feature_length(cls, gene_feature_length_uris: Sequence[str]) -> pd.DataFrame:
        """
        Private. Load any external assets required to create the experiment.
        """
        return pd.concat(
            pd.read_csv(
                io.BytesIO(cat_file(uri)),
                names=["feature_id", "feature_name", "gene_version", "feature_length"],
            )
            .set_index("feature_id")
            .drop(columns=["feature_name", "gene_version"])
            for uri in gene_feature_length_uris
        )


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
        self.var_df: pd.DataFrame = pd.DataFrame(columns=["feature_id", "feature_name"])
        self.dataset_obs_joinid_start: Dict[str, int] = {}
        self.census_summary_cell_counts = init_summary_counts_accumulator()
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

    @property
    def gene_feature_length(self) -> pd.DataFrame:
        return self.specification.gene_feature_length

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
            "obs", schema=obs_schema, index_column_names=["soma_joinid"], platform_config=CENSUS_OBS_PLATFORM_CONFIG
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
        return anndata_cell_filter(ad, retain_X=False)

    def accumulate_axes(self, dataset: Dataset, ad: anndata.AnnData) -> int:
        """
        Write obs, accumulate var.

        Returns: number of cells that make it past the experiment filter.
        """
        assert len(ad.obs) > 0

        # Narrow columns just to minimize memory footprint. Summary cell counting
        # requires 'organism', do be careful not to delete that.
        obs_df = ad.obs[list(CXG_OBS_TERM_COLUMNS) + ["organism"]].reset_index(drop=True).copy()

        obs_df["soma_joinid"] = range(self.n_obs + 1, self.n_obs + len(obs_df) + 1)
        obs_df["dataset_id"] = dataset.dataset_id

        # high-level tissue mapping
        add_tissue_mapping(obs_df, dataset.dataset_id)

        # Accumulate aggregation counts
        self.census_summary_cell_counts = accumulate_summary_counts(self.census_summary_cell_counts, obs_df)

        # drop columns we don't want to write
        obs_df = obs_df[list(CENSUS_OBS_TERM_COLUMNS)]
        obs_df = anndata_ordered_bool_issue_853_workaround(obs_df)

        self.populate_obs_axis(obs_df)

        self.dataset_obs_joinid_start[dataset.dataset_id] = self.n_obs

        # Accumulate the union of all var ids/names (for raw and processed), to be later persisted.
        # NOTE: assumes raw.var is None, OR has same index as var. Currently enforced in open_anndata(),
        # but may need to evolve this logic if that assumption is not scalable.
        tv = ad.var.rename_axis("feature_id").reset_index()[["feature_id", "feature_name"]]
        self.var_df = pd.concat([self.var_df, tv]).drop_duplicates()

        self.n_obs += len(obs_df)
        self.n_unique_obs += (obs_df.is_primary_data == True).sum()  # noqa: E712

        donors = obs_df.donor_id.unique()
        self.n_donors += len(donors) - np.isin(donors, DONOR_ID_IGNORE).sum()

        self.n_datasets += 1
        return len(obs_df)

    def populate_obs_axis(self, obs_df: pd.DataFrame) -> None:
        _assert_open_for_write(self.experiment)

        logging.debug(f"experiment {self.name} obs = {obs_df.shape}")
        pa_table = pa.Table.from_pandas(
            obs_df,
            preserve_index=False,
            columns=list(CENSUS_OBS_TERM_COLUMNS),
        )
        self.experiment.obs.write(pa_table)  # type:ignore

    def populate_var_axis(self) -> None:
        logging.info(f"{self.name}: populate var axis")

        _assert_open_for_write(self.experiment.ms["RNA"].var)  # type:ignore[union-attr]

        # if is possible there is nothing to write
        if len(self.var_df) > 0:
            # persist var
            self.var_df["soma_joinid"] = range(1, len(self.var_df) + 1)
            self.var_df = self.var_df.join(self.gene_feature_length["feature_length"], on="feature_id")
            self.var_df.feature_length.fillna(0, inplace=True)

            self.var_df = anndata_ordered_bool_issue_853_workaround(self.var_df)

            self.experiment.ms["RNA"].var.write(  # type:ignore
                pa.Table.from_pandas(
                    self.var_df,
                    preserve_index=False,
                    columns=list(CENSUS_VAR_TERM_COLUMNS),
                )
            )

            self.global_var_joinids = self.var_df[["feature_id", "soma_joinid"]].set_index("feature_id")

        self.n_var = len(self.var_df)

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

        # SOMA does not currently arrays with a zero length domain, so special case this corner-case
        # where no data has been read for this experiment.
        if len(self.presence) > 0:
            # sanity check
            assert len(self.presence) == self.n_datasets

            max_dataset_joinid = max(d.soma_joinid for d in datasets)

            # LIL is fast way to create spmatrix
            pm = sparse.lil_array((max_dataset_joinid + 1, self.n_var), dtype=bool)
            for dataset_joinid, presence in self.presence.items():
                data, cols = presence
                pm[dataset_joinid, cols] = data

            pm = pm.tocoo()
            pm.eliminate_zeros()
            assert pm.count_nonzero() == pm.nnz
            assert pm.dtype == bool

            fdpm = self.experiment.ms["RNA"].add_new_sparse_ndarray(  # type:ignore
                FEATURE_DATASET_PRESENCE_MATRIX_NAME, type=pa.bool_(), shape=(max_dataset_joinid + 1, self.n_var)
            )
            fdpm.write(pa.SparseCOOTensor.from_scipy(pm))


def _accumulate_all_X_layers(
    assets_path: str,
    dataset: Dataset,
    experiment_builders: List[ExperimentBuilder],
    dataset_obs_joinid_starts: List[Union[None, int]],
    ms_name: str,
    progress: Tuple[int, int],
) -> PresenceResults:
    """
    For this dataset, save all X layer information for each Experiment. This currently
    includes:
        X['raw'] - raw counts

    Also accumulates presence information per dataset.

    This is a helper function for ExperimentBuilder.accumulate_X
    """
    gc.collect()
    logging.debug(f"Loading AnnData for dataset {dataset.dataset_id} ({progress[0]} of {progress[1]})")
    unfiltered_ad = next(open_anndata(assets_path, [dataset]))[1]
    assert unfiltered_ad.isbacked is False

    presence = []
    for eb, dataset_obs_joinid_start in zip(experiment_builders, dataset_obs_joinid_starts):
        if dataset_obs_joinid_start is None:
            # this dataset has no data for this experiment
            continue

        if eb.n_var == 0:
            # edge case for test builds that have no data for an entire experiment (organism)
            continue

        anndata_cell_filter = make_anndata_cell_filter(eb.anndata_cell_filter_spec)
        ad = anndata_cell_filter(unfiltered_ad)
        if ad.n_obs == 0:
            continue

        # follow CELLxGENE 3.0 schema conventions for raw/X aliasing when only raw counts exist
        raw_X, raw_var = (ad.X, ad.var) if ad.raw is None else (ad.raw.X, ad.raw.var)

        if not is_nonnegative_integral(raw_X):
            logging.error(f"{dataset.dataset_id} contains non-integer or negative valued data")

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
            X_remap = sparse.coo_array((X.data, (row, col)), shape=(eb.n_obs, eb.n_var))
            X_remap.eliminate_zeros()
            with soma.Experiment.open(eb.experiment_uri, "w") as experiment:
                experiment.ms[ms_name].X[layer_name].write(pa.SparseCOOTensor.from_scipy(X_remap))
            gc.collect()

        # Save presence information by dataset_id
        assert dataset.soma_joinid > 0  # i.e., it was assigned prior to this step
        pres_data = raw_X.sum(axis=0) > 0
        if isinstance(pres_data, np.matrix):
            pres_data = pres_data.A
        pres_data = pres_data[0]
        pres_cols = local_var_joinids[np.arange(ad.n_vars, dtype=np.int64)]

        assert pres_data.dtype == bool
        assert pres_cols.dtype == np.int64
        assert pres_data.shape == (ad.n_vars,)
        assert pres_data.shape == pres_cols.shape
        assert ad.n_vars <= eb.n_var

        presence.append(
            (
                dataset.dataset_id,
                dataset.soma_joinid,
                eb.name,
                pres_data,
                pres_cols,
            )
        )

    gc.collect()
    return tuple(presence)


@overload
def _accumulate_X(
    assets_path: str, dataset: Dataset, experiment_builders: List["ExperimentBuilder"], progress: Tuple[int, int]
) -> PresenceResults:
    ...


@overload
def _accumulate_X(
    assets_path: str,
    dataset: Dataset,
    experiment_builders: List["ExperimentBuilder"],
    progress: Tuple[int, int],
    executor: Optional[concurrent.futures.Executor],
) -> concurrent.futures.Future[PresenceResults]:
    ...


def _accumulate_X(
    assets_path: str,
    dataset: Dataset,
    experiment_builders: List["ExperimentBuilder"],
    progress: Tuple[int, int],
    executor: Optional[concurrent.futures.Executor] = None,
) -> Union[concurrent.futures.Future[PresenceResults], PresenceResults]:
    """
    Save X layer data for a single AnnData, for all Experiments. Return a future if
    executor is specified, otherwise immediately do the work.
    """
    for eb in experiment_builders:
        # sanity checks
        assert eb.dataset_obs_joinid_start is not None
        # clear the experiment object to avoid pickling error
        eb.experiment = None

    dataset_obs_joinid_starts = [
        eb.dataset_obs_joinid_start.get(dataset.dataset_id, None) for eb in experiment_builders
    ]

    if executor is not None:
        return executor.submit(
            _accumulate_all_X_layers,
            assets_path,
            dataset,
            experiment_builders,
            dataset_obs_joinid_starts,
            "RNA",
            progress,
        )
    else:
        return _accumulate_all_X_layers(
            assets_path, dataset, experiment_builders, dataset_obs_joinid_starts, MEASUREMENT_RNA_NAME, progress
        )


def populate_X_layers(
    assets_path: str, datasets: List[Dataset], experiment_builders: List[ExperimentBuilder], args: CensusBuildArgs
) -> None:
    """
    Do all X layer processing for all Experiments. Also accumulate presence matrix data for later writing.
    """
    # populate X layers
    presence: List[PresenceResult] = []
    if args.config.multi_process:
        with create_process_pool_executor(args) as pe:
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
                presence += f.result()
                logging.info(f"populate X for dataset {futures[f].dataset_id} ({n} of {len(futures)}) complete.")

    else:
        for n, dataset in enumerate(datasets, start=1):
            presence += _accumulate_X(assets_path, dataset, experiment_builders, progress=(n, len(datasets)))

    eb_by_name = {e.name: e for e in experiment_builders}
    for _, dataset_soma_joinid, eb_name, pres_dataset, pres_col in presence:
        eb_by_name[eb_name].presence[dataset_soma_joinid] = (pres_dataset, pres_col)


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
