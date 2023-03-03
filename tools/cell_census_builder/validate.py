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
import pandas as pd
import pyarrow as pa
import tiledb
import tiledbsoma as soma
from scipy import sparse
from typing_extensions import Self

from .anndata import make_anndata_cell_filter, open_anndata
from .consolidate import list_uris_to_consolidate
from .datasets import Dataset
from .experiment_builder import ExperimentSpecification
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
from .mp import create_process_pool_executor, log_on_broken_process_pool
from .util import uricat


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
    return soma.Experiment.open(uricat(base_uri, CENSUS_DATA_NAME, eb.name), mode="r")


def validate_all_soma_objects_exist(soma_path: str, experiment_specifications: List[ExperimentSpecification]) -> bool:
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

    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        assert soma.Collection.exists(census.uri)
        assert census.metadata["cxg_schema_version"] == CXG_SCHEMA_VERSION
        assert census.metadata["census_schema_version"] == CENSUS_SCHEMA_VERSION
        assert datetime.fromisoformat(census.metadata["created_on"])
        assert "git_commit_sha" in census.metadata

        for name in [CENSUS_INFO_NAME, CENSUS_DATA_NAME]:
            assert soma.Collection.exists(census[name].uri)

        census_info = census[CENSUS_INFO_NAME]
        for name in [CENSUS_DATASETS_NAME, CENSUS_SUMMARY_NAME, CENSUS_SUMMARY_CELL_COUNTS_NAME]:
            assert name in census_info, f"`{name}` missing from census_info"
            assert soma.DataFrame.exists(census_info[name].uri)

        assert sorted(census_info[CENSUS_DATASETS_NAME].keys()) == sorted(CENSUS_DATASETS_COLUMNS + ["soma_joinid"])
        assert sorted(census_info[CENSUS_SUMMARY_CELL_COUNTS_NAME].keys()) == sorted(
            list(CENSUS_SUMMARY_CELL_COUNTS_COLUMNS) + ["soma_joinid"]
        )
        assert sorted(census_info[CENSUS_SUMMARY_NAME].keys()) == sorted(["label", "value", "soma_joinid"])

        # verify required dataset fields are set
        df: pd.DataFrame = census_info[CENSUS_DATASETS_NAME].read().concat().to_pandas()
        assert (df["collection_id"] != "").all()
        assert (df["collection_name"] != "").all()
        assert (df["dataset_title"] != "").all()

        # there should be an experiment for each builder
        census_data = census[CENSUS_DATA_NAME]
        for eb in experiment_specifications:
            assert soma.Experiment.exists(census_data[eb.name].uri)

            e = census_data[eb.name]
            assert soma.DataFrame.exists(e.obs.uri)
            assert soma.Collection.exists(e.ms.uri)

            # there should be a single measurement called 'RNA'
            assert soma.Measurement.exists(e.ms["RNA"].uri)

            # The measurement should contain all X layers where n_obs > 0 (existence checked elsewhere)
            rna = e.ms["RNA"]
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

        return True


def _validate_axis_dataframes(args: Tuple[str, str, Dataset, List[ExperimentSpecification]]) -> Dict[str, EbInfo]:
    assets_path, soma_path, dataset, experiment_specifications = args
    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        census_data = census[CENSUS_DATA_NAME]
        dataset_id = dataset.dataset_id
        _, unfiltered_ad = next(open_anndata(assets_path, [dataset], backed="r"))
        eb_info: Dict[str, EbInfo] = {}
        for eb in experiment_specifications:
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
    experiment_specifications: List[ExperimentSpecification],
    args: argparse.Namespace,
) -> Dict[str, EbInfo]:
    """ "
    Validate axis dataframes: schema, shape, contents

    Raises on error.  Returns True on success.
    """
    logging.debug("validate_axis_dataframes")
    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        census_data = census[CENSUS_DATA_NAME]

        # check schema
        expected_obs_columns = CENSUS_OBS_TERM_COLUMNS
        expected_var_columns = CENSUS_VAR_TERM_COLUMNS
        for eb in experiment_specifications:
            obs = census_data[eb.name].obs
            var = census_data[eb.name].ms["RNA"].var
            assert sorted(obs.keys()) == sorted(expected_obs_columns.keys())
            assert sorted(var.keys()) == sorted(expected_var_columns.keys())
            for field in obs.schema:
                assert field.type == expected_obs_columns[field.name], f"Unexpected type in {field.name}: {field.type}"
            for field in var.schema:
                assert field.type == expected_var_columns[field.name], f"Unexpected type in {field.name}: {field.type}"

    # check shapes & perform weak test of contents
    eb_info = {eb.name: EbInfo() for eb in experiment_specifications}
    if args.multi_process:
        with create_process_pool_executor(args) as ppe:
            futures = [
                ppe.submit(_validate_axis_dataframes, (assets_path, soma_path, dataset, experiment_specifications))
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
                (assets_path, soma_path, dataset, experiment_specifications)
            ).items():
                eb_info[eb_name].update(ebi)
            logging.info(f"validate_axis {n} of {len(datasets)} complete.")

    for eb in experiment_specifications:
        with open_experiment(soma_path, eb) as exp:
            n_vars = len(eb_info[eb.name].vars)

            census_obs_df = exp.obs.read(column_names=["soma_joinid", "dataset_id"]).concat().to_pandas()
            assert eb_info[eb.name].n_obs == len(census_obs_df)
            assert (len(census_obs_df) == 0) or (census_obs_df.soma_joinid.max() + 1 == eb_info[eb.name].n_obs)
            assert eb_info[eb.name].dataset_ids == set(census_obs_df.dataset_id.unique())

            census_var_df = exp.ms["RNA"].var.read(column_names=["feature_id", "soma_joinid"]).concat().to_pandas()
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


def _validate_X_layers_contents_by_dataset(args: Tuple[str, str, Dataset, List[ExperimentSpecification]]) -> bool:
    """
    Validate that a single dataset is correctly represented in the census.
    Intended to be dispatched from validate_X_layers.

    Currently, implements weak tests:
    * the nnz is correct
    * there are no zeros explicitly saved (this is mandated by cell census schema)
    """
    assets_path, soma_path, dataset, experiment_specifications = args
    _, unfiltered_ad = next(open_anndata(assets_path, [dataset]))

    for eb in experiment_specifications:
        with open_experiment(soma_path, eb) as exp:
            anndata_cell_filter = make_anndata_cell_filter(eb.anndata_cell_filter_spec)
            ad = anndata_cell_filter(unfiltered_ad, retain_X=True)

            soma_joinids: npt.NDArray[np.int64] = (
                exp.obs.read(
                    column_names=["soma_joinid", "dataset_id"], value_filter=f"dataset_id == '{dataset.dataset_id}'"
                )
                .concat()
                .to_pandas()
                .soma_joinid.to_numpy()
            )

            raw_nnz = 0
            if len(soma_joinids) > 0:
                assert soma.SparseNDArray.exists(exp.ms["RNA"].X["raw"].uri)

                def count_elements(arr: soma.SparseNDArray, join_ids: npt.NDArray[np.int64]) -> int:
                    return sum(t.non_zero_length for t in arr.read((join_ids, slice(None))).coos())

                raw_nnz = count_elements(exp.ms["RNA"].X["raw"], soma_joinids)

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


def _validate_X_layer_has_unique_coords(args: Tuple[ExperimentSpecification, str, str, int, int]) -> bool:
    """Validate that all X layers have no duplicate coordinates"""
    experiment_specification, soma_path, layer_name, row_range_start, row_range_stop = args
    with open_experiment(soma_path, experiment_specification) as exp:
        logging.info(
            f"validate_no_dups_X start, {experiment_specification.name}, {layer_name}, rows [{row_range_start}, {row_range_stop})"
        )
        if layer_name not in exp.ms["RNA"].X:
            return True

        X_layer = exp.ms["RNA"].X[layer_name]
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
    experiment_specifications: List[ExperimentSpecification],
    eb_info: Dict[str, EbInfo],
    args: argparse.Namespace,
) -> bool:
    """ "
    Validate all X layers: schema, shape, contents

    Raises on error.  Returns True on success.
    """
    logging.debug("validate_X_layers")
    for eb in experiment_specifications:
        with open_experiment(soma_path, eb) as exp:
            assert soma.Collection.exists(exp.ms["RNA"].X.uri)

            census_obs_df = exp.obs.read(column_names=["soma_joinid"]).concat().to_pandas()
            n_obs = len(census_obs_df)
            logging.info(f"uri = {exp.obs.uri}, eb.n_obs = {eb_info[eb.name].n_obs}; n_obs = {n_obs}")
            assert n_obs == eb_info[eb.name].n_obs

            census_var_df = exp.ms["RNA"].var.read(column_names=["feature_id", "soma_joinid"]).concat().to_pandas()
            n_vars = len(census_var_df)
            assert n_vars == eb_info[eb.name].n_vars

            if n_obs > 0:
                for lyr in CENSUS_X_LAYERS:
                    assert soma.SparseNDArray.exists(exp.ms["RNA"].X[lyr].uri)
                    X = exp.ms["RNA"].X[lyr]
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
                    (eb, soma_path, layer_name, row_start, row_start + ROWS_PER_PROCESS),
                )
                for eb in experiment_specifications
                for layer_name in CENSUS_X_LAYERS
                for row_start in range(0, n_obs, ROWS_PER_PROCESS)
            ]
            per_dataset_futures = [
                ppe.submit(
                    _validate_X_layers_contents_by_dataset, (assets_path, soma_path, dataset, experiment_specifications)
                )
                for dataset in datasets
            ]
            for n, future in enumerate(concurrent.futures.as_completed(dup_coord_futures), start=1):
                log_on_broken_process_pool(ppe)
                assert future.result()
                logging.info(f"validate_no_dups_X {n} of {len(dup_coord_futures)} complete.")
            for n, future in enumerate(concurrent.futures.as_completed(per_dataset_futures), start=1):
                log_on_broken_process_pool(ppe)
                assert future.result()
                logging.info(f"validate_X {n} of {len(datasets)} complete.")

    else:
        for eb in experiment_specifications:
            for layer_name in CENSUS_X_LAYERS:
                logging.info(f"Validating no duplicate coordinates in X layer {eb.name} layer {layer_name}")
                assert _validate_X_layer_has_unique_coords((eb, soma_path, layer_name, 0, n_obs))
        for n, vld in enumerate(
            (
                _validate_X_layers_contents_by_dataset((assets_path, soma_path, dataset, experiment_specifications))
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
    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        df = census[CENSUS_INFO_NAME][CENSUS_DATASETS_NAME].read().concat().to_pandas()
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


def validate_consolidation(soma_path: str, experiment_specifications: List[ExperimentSpecification]) -> bool:
    """Verify that obs, var and X layers are all fully consolidated & vacuumed"""

    def is_empty_tiledb_array(uri: str) -> bool:
        with tiledb.open(uri) as A:
            return A.nonempty_domain() is None

    with soma.Collection.open(soma_path, context=SOMA_TileDB_Context()) as census:
        consolidated_uris = list_uris_to_consolidate(census)
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
    assert os.path.exists(soma_path) and os.path.exists(assets_path)
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


def validate(
    args: argparse.Namespace, soma_path: str, assets_path: str, experiment_specifications: List[ExperimentSpecification]
) -> bool:
    """
    Validate that the "census" matches the datasets and experiment builder spec.

    Will raise if validation fails. Returns True on success.
    """
    logging.info("Validation start")
    assert validate_directory_structure(soma_path, assets_path)

    assert soma_path.endswith("soma") and assets_path.endswith("h5ads")
    assert validate_all_soma_objects_exist(soma_path, experiment_specifications)
    assert validate_relative_path(soma_path)
    datasets = load_datasets_from_census(assets_path, soma_path)
    assert validate_manifest_contents(assets_path, datasets)

    assert (eb_info := validate_axis_dataframes(assets_path, soma_path, datasets, experiment_specifications, args))
    assert validate_X_layers(assets_path, soma_path, datasets, experiment_specifications, eb_info, args)
    assert validate_consolidation(soma_path, experiment_specifications)
    logging.info("Validation finished (success)")
    return True
