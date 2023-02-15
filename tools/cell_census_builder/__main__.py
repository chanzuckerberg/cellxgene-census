import argparse
import gc
import logging
import multiprocessing
import os.path
import sys
from datetime import datetime, timezone
from typing import List, Union

import pyarrow as pa
import tiledbsoma as soma
from anndata import AnnData

from .anndata import open_anndata
from .census_summary import create_census_summary
from .consolidate import consolidate
from .datasets import Dataset, assign_dataset_soma_joinids, create_dataset_manifest
from .experiment_builder import ExperimentBuilder, populate_X_layers, reopen_experiment_builders
from .globals import (
    CENSUS_DATA_NAME,
    CENSUS_INFO_NAME,
    CENSUS_SCHEMA_VERSION,
    CXG_SCHEMA_VERSION,
    FEATURE_DATASET_PRESENCE_MATRIX_NAME,
    RNA_SEQ,
    SOMA_TileDB_Context,
)
from .manifest import load_manifest
from .mp import process_initializer
from .source_assets import stage_source_assets
from .summary_cell_counts import create_census_summary_cell_counts
from .util import get_git_commit_sha, is_git_repo_dirty, uricat
from .validate import validate


def make_experiment_builders() -> List[ExperimentBuilder]:
    """
    Define all soma.Experiments to build in the census.

    Functionally, this defines per-experiment name, anndata filter, etc.
    It also loads any required per-Experiment assets.
    """
    GENE_LENGTH_BASE_URI = (
        "https://raw.githubusercontent.com/chanzuckerberg/single-cell-curation/"
        "100f935eac932e1f5f5dadac0627204da3790f6f/cellxgene_schema_cli/cellxgene_schema/ontology_files/"
    )
    GENE_LENGTH_URIS = [
        GENE_LENGTH_BASE_URI + "genes_homo_sapiens.csv.gz",
        GENE_LENGTH_BASE_URI + "genes_mus_musculus.csv.gz",
        GENE_LENGTH_BASE_URI + "genes_sars_cov_2.csv.gz",
    ]
    experiment_builders = [  # The soma.Experiments we want to build
        ExperimentBuilder(
            name="homo_sapiens",
            anndata_cell_filter_spec=dict(organism_ontology_term_id="NCBITaxon:9606", assay_ontology_term_ids=RNA_SEQ),
            gene_feature_length_uris=GENE_LENGTH_URIS,
        ),
        ExperimentBuilder(
            name="mus_musculus",
            anndata_cell_filter_spec=dict(organism_ontology_term_id="NCBITaxon:10090", assay_ontology_term_ids=RNA_SEQ),
            gene_feature_length_uris=GENE_LENGTH_URIS,
        ),
    ]

    return experiment_builders


def main() -> int:
    parser = create_args_parser()
    args = parser.parse_args()
    assert args.subcommand in ["build", "validate"]

    process_initializer(args.verbose)

    # normalize our base URI - must include trailing slash
    soma_path = uricat(args.uri, args.build_tag, "soma")
    assets_path = uricat(args.uri, args.build_tag, "h5ads")

    # create the experiment builders
    experiment_builders = make_experiment_builders()

    cc = 0
    if args.subcommand == "build":
        cc = build(args, soma_path, assets_path, experiment_builders)

    if cc == 0 and (args.subcommand == "validate" or args.validate):
        validate(args, soma_path, assets_path, experiment_builders)

    return cc


def prepare_file_system(soma_path: str, assets_path: str, args: argparse.Namespace) -> None:
    """
    Prepares the file system for the builder run
    """
    # Don't clobber an existing census build
    if os.path.exists(soma_path) or os.path.exists(assets_path):
        raise Exception("Census build path already exists - aborting build")

    # Ensure that the git tree is clean
    if not args.test_disable_dirty_git_check and is_git_repo_dirty():
        raise Exception("The git repo has uncommitted changes - aborting build")

    # Create top-level build directories
    os.makedirs(soma_path, exist_ok=False)
    os.makedirs(assets_path, exist_ok=False)


def build(
    args: argparse.Namespace, soma_path: str, assets_path: str, experiment_builders: List[ExperimentBuilder]
) -> int:
    """
    Approximately, build steps are:
    1. Download manifest and copy/stage all source assets
    2. Read all H5AD and create axis dataframe (serial)
        * write obs/var dataframes
        * accumulate overall shape of X
    3. Read all H5AD assets again, write X layer (parallel)
    4. Optional: validate

    Returns
    -------
    int
        Process completion code, 0 on success, non-zero indicating error,
        suitable for providing to sys.exit()
    """

    try:
        prepare_file_system(soma_path, assets_path, args)
    except Exception as e:
        logging.error(e)
        return 1

    # Step 1 - get all source datasets
    datasets = build_step1_get_source_datasets(args, assets_path)

    # Step 2 - create root collection, and all child objects, but do not populate any dataframes or matrices
    root_collection = build_step2_create_root_collection(soma_path, experiment_builders)
    gc.collect()

    # Step 3 - populate axes
    filtered_datasets = build_step3_populate_obs_and_var_axes(assets_path, datasets, experiment_builders)

    # Step 4 - populate axes and X layers
    build_step4_populate_X_layers(assets_path, filtered_datasets, experiment_builders, args)
    gc.collect()

    # Step 5- write out dataset manifest and summary information
    build_step5_populate_summary_info(root_collection, experiment_builders, filtered_datasets, args.build_tag)

    for eb in experiment_builders:
        eb.build_completed = True

    if args.consolidate:
        consolidate(args, root_collection.uri)

    return 0


def populate_root_collection(root_collection: soma.Collection) -> soma.Collection:
    """
    Create the root SOMA collection for the Census.

    Returns the root collection.
    """

    # Set root metadata for the experiment
    root_collection.metadata["created_on"] = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    root_collection.metadata["cxg_schema_version"] = CXG_SCHEMA_VERSION
    root_collection.metadata["census_schema_version"] = CENSUS_SCHEMA_VERSION

    sha = get_git_commit_sha()
    root_collection.metadata["git_commit_sha"] = sha

    # Create sub-collections for experiments, etc.
    for n in [CENSUS_INFO_NAME, CENSUS_DATA_NAME]:
        root_collection.add_new_collection(n)

    return root_collection


def build_step1_get_source_datasets(args: argparse.Namespace, assets_path: str) -> List[Dataset]:
    logging.info("Build step 1 - get source assets - started")

    # Load manifest defining the datasets
    datasets = load_manifest(args.manifest)
    if len(datasets) == 0:
        logging.error("No H5AD files in the manifest (or we can't find the files)")
        raise AssertionError("No H5AD files in the manifest (or we can't find the files)")

    # Testing/debugging hook - hidden option
    if args.test_first_n is not None and args.test_first_n > 0:
        # Process the N smallest datasets
        datasets = sorted(datasets, key=lambda d: d.asset_h5ad_filesize)[0 : args.test_first_n]

    # Stage all files
    stage_source_assets(datasets, args, assets_path)

    logging.info("Build step 1 - get source assets - finished")
    return datasets


def populate_obs_axis(
    assets_path: str, datasets: List[Dataset], experiment_builders: List[ExperimentBuilder]
) -> List[Dataset]:
    filtered_datasets = []
    N = len(datasets) * len(experiment_builders)
    n = 0

    for dataset, ad in open_anndata(assets_path, datasets, backed="r"):
        dataset_total_cell_count = 0

        for eb in reopen_experiment_builders(experiment_builders):
            n += 1
            logging.info(f"{eb.name}: filtering dataset '{dataset.dataset_id}' ({n} of {N})")
            ad_filtered = eb.filter_anndata_cells(ad)

            if len(ad_filtered.obs) == 0:
                logging.info(f"{eb.name} - H5AD has no data after filtering, skipping {dataset.dataset_h5ad_path}")
                continue

            # append to `obs`; accumulate `var` data
            dataset_total_cell_count += eb.accumulate_axes(dataset, ad_filtered)

        # dataset passes filter if either experiment includes cells from the dataset
        if dataset_total_cell_count > 0:
            filtered_datasets.append(dataset)
            dataset.dataset_total_cell_count = dataset_total_cell_count

    for eb in experiment_builders:
        logging.info(f"Experiment {eb.name} will contain {eb.n_obs} cells from {eb.n_datasets} datasets")




    return filtered_datasets


def populate_var_axis_and_presence(experiment_builders: List[ExperimentBuilder]) -> int:
    for eb in reopen_experiment_builders(experiment_builders):
        # populate `var`; create empty `presence` now that we have its dimensions
        eb.populate_var_axis()

        # SOMA does not currently support empty arrays, so special case this corner-case.
        if eb.n_var > 0:
            eb.experiment.ms["RNA"].add_new_sparse_ndarray(
                FEATURE_DATASET_PRESENCE_MATRIX_NAME, type=pa.bool_(), shape=(eb.n_datasets + 1, eb.n_var)
            )


def build_step2_create_root_collection(
    soma_path: str, experiment_builders: List[ExperimentBuilder]
) -> soma.Collection:
    """
    Create all objects

    Returns: the root collection.
    """
    logging.info("Build step 2 - axis creation - started")

    with soma.Collection.create(soma_path, context=SOMA_TileDB_Context()) as root_collection:
        populate_root_collection(root_collection)

        for e in experiment_builders:
            e.create(census_data=root_collection[CENSUS_DATA_NAME])

        logging.info("Build step 2 - axis creation - finished")
        return root_collection


def build_step3_populate_obs_and_var_axes(
        assets_path: str,
        datasets: List[Dataset],
        experiment_builders: List[ExperimentBuilder],
    ) -> List[Dataset]:
    """
    Populate obs and var axes. Filter cells from datasets for each experiment, as obs is built.
    """
    logging.info("Build step 3 - Populate obs and var axes - started")

    filtered_datasets = populate_obs_axis(assets_path, datasets, experiment_builders)
    logging.info(f"({len(filtered_datasets)} of {len(datasets)}) datasets suitable for processing.")

    populate_var_axis_and_presence(experiment_builders)

    assign_dataset_soma_joinids(filtered_datasets)

    logging.info("Build step 3 - Populate obs and var axes - finished")

    return filtered_datasets


def build_step4_populate_X_layers(
    assets_path: str,
    filtered_datasets: List[Dataset],
    experiment_builders: List[ExperimentBuilder],
    args: argparse.Namespace,
) -> None:
    """
    Populate X layers.
    """
    logging.info("Build step 3 - populate X layers - started")

    # Process all X data
    for eb in reopen_experiment_builders(experiment_builders):
        eb.create_X_with_layers()

    populate_X_layers(assets_path, filtered_datasets, experiment_builders, args)

    for eb in reopen_experiment_builders(experiment_builders):
        eb.populate_presence_matrix(filtered_datasets)

    logging.info("Build step 3 - populate X layers - finished")


def build_step5_populate_summary_info(
    root_collection: soma.Collection,
    experiment_builders: List[ExperimentBuilder],
    filtered_datasets: List[Dataset],
    build_tag: str,
) -> None:
    logging.info("Build step 4 - summary info - started")

    with soma.Collection.open(root_collection[CENSUS_INFO_NAME].uri, "w", context=SOMA_TileDB_Context()) as census_info:
        create_dataset_manifest(census_info, filtered_datasets)
        create_census_summary_cell_counts(census_info, [e.census_summary_cell_counts for e in experiment_builders])
        create_census_summary(census_info, experiment_builders, build_tag)

    logging.info("Build step 4 - summary info - finished")


def create_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cell_census_builder")
    parser.add_argument("uri", type=str, help="Census top-level URI")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity")
    parser.add_argument(
        "-mp",
        "--multi-process",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use multiple processes",
    )
    parser.add_argument("--max-workers", type=int, help="Concurrency")
    parser.add_argument(
        "--build-tag",
        type=str,
        default=datetime.now().astimezone().date().isoformat(),
        help="Census build tag (default: current date is ISO8601 format)",
    )

    subparsers = parser.add_subparsers(required=True, dest="subcommand")

    # BUILD
    build_parser = subparsers.add_parser("build", help="Build Cell Census")
    build_parser.add_argument(
        "--manifest",
        type=argparse.FileType("r"),
        help="Manifest file",
    )
    build_parser.add_argument(
        "--validate", action=argparse.BooleanOptionalAction, default=True, help="Validate immediately after build"
    )
    build_parser.add_argument(
        "--consolidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Consolidate TileDB objects after build",
    )
    # hidden option for testing by devs. Will process only the first 'n' datasets
    build_parser.add_argument("--test-first-n", type=int)
    # hidden option for testing by devs. Allow for WIP testing by devs.
    build_parser.add_argument("--test-disable-dirty-git-check", action=argparse.BooleanOptionalAction)

    # VALIDATE
    subparsers.add_parser("validate", help="Validate an existing cell census build")

    return parser


if __name__ == "__main__":
    # this is very important to do early, before any use of `concurrent.futures`
    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)

    sys.exit(main())
