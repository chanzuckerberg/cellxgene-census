import argparse
import concurrent.futures
import gc
import io
import logging
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict, Union, overload

import anndata
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import tiledbsoma as soma
from scipy import sparse
from somacore.experiment import Experiment
from tiledbsoma.tiledb_object import TileDBObject

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
from .mp import create_process_pool_executor
from .source_assets import cat_file
from .summary_cell_counts import accumulate_summary_counts, init_summary_counts_accumulator
from .tissue_mapper import TissueMapper  # type: ignore
from .util import (
    anndata_ordered_bool_issue_853_workaround,
    array_chunker,
    is_positive_integral,
    uricat,
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


def _assert_open_for_write(obj: TileDBObject) -> None:
    assert obj is not None
    assert obj.exists(obj.uri)
    assert obj.mode == "w"
    assert not obj.closed


class ExperimentBuilder:
    """
    Class to help build a parameterized SOMA experiment, where key parameters are:
    * experiment "name" (eg, 'human'), must be unique in all experiments.
    * an AnnData filter used to cherry pick data for the experiment
    * methods to progressively build the experiment

    The creation and driving of these objects is done by the main loop.
    """

    name: str
    anndata_cell_filter_spec: AnnDataFilterSpec
    gene_feature_length_uris: List[str]
    gene_feature_length: pd.DataFrame

    def __init__(self, name: str, anndata_cell_filter_spec: AnnDataFilterSpec, gene_feature_length_uris: List[str]):
        self.name = name
        self.anndata_cell_filter_spec = anndata_cell_filter_spec
        self.gene_feature_length_uris = gene_feature_length_uris

        # accumulated state
        self.n_obs: int = 0
        self.n_unique_obs: int = 0
        self.n_var: int = 0
        self.n_datasets: int = 0
        self.n_donors: int = 0  # Caution: defined as (unique dataset_id, donor_id) tuples, *excluding* some values
        self.var_df: pd.DataFrame = pd.DataFrame(columns=["feature_id", "feature_name"])
        self.obs_df: pd.DataFrame = pd.DataFrame(columns=CENSUS_OBS_TERM_COLUMNS)
        self.dataset_obs_joinid_start: Optional[Dict[str, int]] = None  # initialized in create_joinid_metadata()
        self.census_summary_cell_counts = init_summary_counts_accumulator()
        self.experiment: Optional[soma.Experiment] = None  # initialized in create()
        self.global_var_joinids: Optional[pd.DataFrame] = None
        self.presence: Dict[int, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]] = {}

        self.load_assets()

    def load_assets(self) -> None:
        """
        Load any external assets required to create the experiment.
        """
        self.gene_feature_length = pd.concat(
            pd.read_csv(
                io.BytesIO(cat_file(uri)),
                names=["feature_id", "feature_name", "gene_version", "feature_length"],
            )
            .set_index("feature_id")
            .drop(columns=["feature_name", "gene_version"])
            for uri in self.gene_feature_length_uris
        )
        logging.info(f"Loaded gene lengths external reference for {self.name}, {len(self.gene_feature_length)} genes.")

    def create(self, census_data: soma.Collection, filtered_datasets: List[Dataset]) -> None:
        """Create experiment within the specified Collection with a single Measurement."""

        logging.info(f"{self.name}: create experiment at {uricat(census_data.uri, self.name)}")

        self.experiment = census_data.add_new_collection(self.name, soma.Experiment)

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

        # SOMA does not currently support empty arrays, so special case this corner-case.
        if self.n_var > 0:
            max_dataset_joinid = max(d.soma_joinid for d in filtered_datasets)

            rna_measurement.add_new_sparse_ndarray(
                FEATURE_DATASET_PRESENCE_MATRIX_NAME, type=pa.bool_(), shape=(max_dataset_joinid + 1, self.n_var)
            )

        self.populate_obs_axis()
        self.populate_var_axis()
        self.create_X_with_layers()
        self.create_joinid_metadata()

    def accumulate_axes(self, dataset: Dataset, ad: anndata.AnnData, progress: Tuple[int, int] = (0, 0)) -> int:
        """
        Write obs, accumulate var.

        Returns: number of cells that make it past the experiment filter.
        """

        progmsg = f"({progress[0]} of {progress[1]})"
        logging.info(f"{self.name}: accumulate axis for dataset '{dataset.dataset_id}' {progmsg}")

        anndata_cell_filter = make_anndata_cell_filter(self.anndata_cell_filter_spec)
        ad = anndata_cell_filter(ad, retain_X=False)
        if ad.n_obs == 0:
            logging.info(f"{self.name} - H5AD has no data after filtering, skipping {dataset.dataset_h5ad_path}")
            return 0

        # Narrow columns just to minimize memory footprint. Summary cell counting
        # requires 'organism', do be careful not to delete that.
        obs_df = ad.obs[list(CXG_OBS_TERM_COLUMNS) + ["organism"]].reset_index(drop=True).copy()

        obs_df["soma_joinid"] = range(self.n_obs, self.n_obs + len(obs_df))
        obs_df["dataset_id"] = dataset.dataset_id

        # high-level tissue mapping
        add_tissue_mapping(obs_df, dataset.dataset_id)

        # Accumulate aggregation counts
        self._accumulate_summary_cell_counts(obs_df)

        # drop columns we don't want to write
        obs_df = obs_df[list(CENSUS_OBS_TERM_COLUMNS)]
        self.obs_df = anndata_ordered_bool_issue_853_workaround(obs_df)

        # Accumulate the union of all var ids/names (for raw and processed), to be later persisted.
        # NOTE: assumes raw.var is None, OR has same index as var. Currently enforced in open_anndata(),
        # but may need to evolve this logic if that assumption is not scalable.
        tv = ad.var.rename_axis("feature_id").reset_index()[["feature_id", "feature_name"]]
        self.var_df = pd.concat([self.var_df, tv]).drop_duplicates()

        self.n_obs += len(self.obs_df)
        self.n_unique_obs += (self.obs_df.is_primary_data == True).sum()  # noqa: E712

        donors = obs_df.donor_id.unique()
        self.n_donors += len(donors) - np.isin(donors, DONOR_ID_IGNORE).sum()

        self.n_datasets += 1
        return len(obs_df)

    def populate_obs_axis(self) -> None:
        logging.info(f"experiment {self.name} obs = {self.obs_df.shape}")
        pa_table = pa.Table.from_pandas(
            self.obs_df,
            preserve_index=False,
            columns=list(CENSUS_OBS_TERM_COLUMNS),
        )
        self.experiment.obs.write(pa_table)

    def populate_var_axis(self) -> None:
        logging.info(f"{self.name}: populate var axis")

        _assert_open_for_write(self.experiment.ms["RNA"].var)

        # if is possible there is nothing to write
        if len(self.var_df) > 0:
            # persist var
            self.var_df["soma_joinid"] = range(len(self.var_df))
            self.var_df = self.var_df.join(self.gene_feature_length["feature_length"], on="feature_id")
            self.var_df.feature_length.fillna(0, inplace=True)

            self.var_df = anndata_ordered_bool_issue_853_workaround(self.var_df)

            self.experiment.ms["RNA"].var.write(
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

        rna_measurement = self.experiment.ms[MEASUREMENT_RNA_NAME]
        _assert_open_for_write(rna_measurement)

        # make the `X` collection (but not the actual layers)
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

    def create_joinid_metadata(self) -> None:
        logging.info(f"{self.name}: make joinid metadata")
        assert self.obs_df is not None

        # Map of dataset_id -> starting soma_joinid for obs axis.  This code _assumes_
        # that soma_joinid is contiguous (ie, no deletions in obs), which is
        # known true for our use case (aggregating h5ads).
        self.dataset_obs_joinid_start = (
            self.obs_df[["dataset_id", "soma_joinid"]].groupby("dataset_id").min().soma_joinid.to_dict()
        )

    def _accumulate_summary_cell_counts(self, obs_df: pd.DataFrame) -> None:
        """
        Add summary counts to the census_summary_cell_counts dataframe
        """
        assert "dataset_id" in obs_df
        assert len(obs_df) > 0
        self.census_summary_cell_counts = accumulate_summary_counts(self.census_summary_cell_counts, obs_df)

    def populate_presence_matrix(self, datasets: List[Dataset]) -> None:
        """
        Save presence matrix per Experiment
        """
        _assert_open_for_write(self.experiment)

        if len(self.presence) > 0:
            max_dataset_joinid = max(d.soma_joinid for d in datasets)

            # A few sanity checks
            assert len(self.presence) == self.n_datasets
            assert max_dataset_joinid >= max(self.presence.keys())  # key is dataset joinid

            # LIL is fast way to create spmatrix
            pm = sparse.lil_array((max_dataset_joinid + 1, self.n_var), dtype=bool)
            for dataset_joinid, presence in self.presence.items():
                data, cols = presence
                pm[dataset_joinid, cols] = data

            pm = pm.tocoo()
            pm.eliminate_zeros()
            assert pm.count_nonzero() == pm.nnz
            assert pm.dtype == bool
            fdpm: soma.SparseNDArray = self.experiment.ms[MEASUREMENT_RNA_NAME][FEATURE_DATASET_PRESENCE_MATRIX_NAME]
            fdpm.write(pa.SparseCOOTensor.from_scipy(pm))

    def reopen_for_write(self) -> Experiment:
        assert self.experiment.closed
        self.experiment = soma.Experiment.open(self.experiment.uri, "w", context=SOMA_TileDB_Context())
        return self.experiment


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

        if not is_positive_integral(raw_X):
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
            eb.experiment.ms[ms_name].X[layer_name].write(pa.SparseCOOTensor.from_scipy(X_remap))
            gc.collect()

        # Save presence information by dataset_id
        assert dataset.soma_joinid >= 0  # i.e., it was assigned prior to this step
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
        _assert_open_for_write(eb.experiment)
        assert eb.dataset_obs_joinid_start is not None

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
    assets_path: str, datasets: List[Dataset], experiment_builders: List[ExperimentBuilder], args: argparse.Namespace
) -> None:
    """
    Do all X layer processing for all Experiments.
    """

    # populate X layers
    presence: List[PresenceResult] = []
    if args.multi_process:
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
                # propagate exceptions - not expecting any other return values
                presence += f.result()
                logging.info(f"pass 2, {futures[f].dataset_id} ({n} of {len(futures)}) complete.")

    else:
        for n, dataset in enumerate(datasets, start=1):
            presence += _accumulate_X(assets_path, dataset, experiment_builders, progress=(n, len(datasets)))

    eb_by_name = {e.name: e for e in experiment_builders}
    for _, dataset_soma_joinid, eb_name, pres_data, pres_col in presence:
        eb_by_name[eb_name].presence[dataset_soma_joinid] = (pres_data, pres_col)


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
