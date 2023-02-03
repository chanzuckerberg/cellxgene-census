import argparse
import gc
import logging
import multiprocessing
import os.path
import sys
from datetime import datetime, timezone
from typing import List, Tuple

import tiledbsoma as soma

from .anndata import open_anndata
from .census_summary import create_census_summary
from .consolidate import consolidate
from .datasets import Dataset, assign_soma_joinids, create_dataset_manifest
from .experiment_builder import ExperimentBuilder, populate_X_layers
from .globals import CENSUS_BUILDER_GIT_SHA, CENSUS_SCHEMA_VERSION, CXG_SCHEMA_VERSION, RNA_SEQ, CENSUS_DATA_NAME, CENSUS_INFO_NAME, \
    SOMA_TileDB_Context
from .manifest import load_manifest
from .mp import process_initializer
from .source_assets import stage_source_assets
from .summary_cell_counts import create_census_summary_cell_counts
from .util import get_git_commit_sha, uricat
from .validate import validate


def make_experiment_builders(base_uri: str, args: argparse.Namespace) -> List[ExperimentBuilder]:
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
            base_uri=base_uri,
            name="homo_sapiens",
            anndata_cell_filter_spec=dict(organism_ontology_term_id="NCBITaxon:9606", assay_ontology_term_ids=RNA_SEQ),
            gene_feature_length_uris=GENE_LENGTH_URIS,
        ),
        ExperimentBuilder(
            base_uri=base_uri,
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
    args.uri = args.uri if args.uri.endswith("/") else args.uri + "/"
    soma_path = uricat(args.uri, args.build_tag, "soma")
    assets_path = uricat(args.uri, args.build_tag, "h5ads")

    # create the experiment builders
    experiment_builders = make_experiment_builders(uricat(soma_path, CENSUS_DATA_NAME), args)

    cc = 0
    if args.subcommand == "build":
        cc = build(args, soma_path, assets_path, experiment_builders)

        # sanity check for build completion
        assert cc != 0 or all(e.is_finished() for e in experiment_builders)

    if cc == 0 and (args.subcommand == "validate" or args.validate):
        validate(args, experiment_builders)

    return cc


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

    # Don't clobber an existing census build
    if os.path.exists(soma_path) or os.path.exists(assets_path):
        logging.error("Census build path already exists - aborting build")
        return 1

    # Create top-level build directories
    os.makedirs(soma_path, exist_ok=False)
    os.makedirs(assets_path, exist_ok=False)

    # Step 1 - get all source assets
    datasets = build_step1_get_source_assets(args, assets_path)

    # Step 2 - build axis dataframes
    top_level_collection, filtered_datasets = build_step2_create_axis(
        soma_path, assets_path, datasets, experiment_builders, args
    )
    assign_soma_joinids(filtered_datasets)
    logging.info(f"({len(filtered_datasets)} of {len(datasets)}) suitable for processing.")
    gc.collect()

    # Step 3- create X layers
    build_step3_create_X_layers(assets_path, filtered_datasets, experiment_builders, args)
    gc.collect()

    # Write out dataset manifest and summary information
    create_dataset_manifest(top_level_collection[CENSUS_INFO_NAME], filtered_datasets)
    create_census_summary_cell_counts(
        top_level_collection[CENSUS_INFO_NAME], [e.census_summary_cell_counts for e in experiment_builders]
    )
    create_census_summary(top_level_collection[CENSUS_INFO_NAME], experiment_builders, args.build_tag)

    if args.consolidate:
        consolidate(args, top_level_collection.uri)

    add_git_commit_sha(top_level_collection)

    return 0


def create_top_level_collections(soma_path: str) -> soma.Collection:
    """
    Create the top-level SOMA collections for the Census.

    Returns the top-most collection.
    """
    top_level_collection = soma.Collection(soma_path, context=SOMA_TileDB_Context())
    if top_level_collection.exists():
        logging.error("Census already exists - aborting")
        raise Exception("Census already exists - aborting")

    top_level_collection.create()
    # Set top-level metadata for the experiment
    top_level_collection.metadata["created_on"] = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    top_level_collection.metadata["cxg_schema_version"] = CXG_SCHEMA_VERSION
    top_level_collection.metadata["census_schema_version"] = CENSUS_SCHEMA_VERSION

    # Create sub-collections for experiments, etc.
    for n in [CENSUS_INFO_NAME, CENSUS_DATA_NAME]:
        cltn = soma.Collection(uricat(top_level_collection.uri, n),
                               context=SOMA_TileDB_Context()).create()
        top_level_collection.set(n, cltn, relative=True)

    return top_level_collection


def build_step1_get_source_assets(args: argparse.Namespace, assets_path: str) -> List[Dataset]:
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


def build_step2_create_axis(
    soma_path: str,
    assets_path: str,
    datasets: List[Dataset],
    experiment_builders: List[ExperimentBuilder],
    args: argparse.Namespace,
) -> Tuple[soma.Collection, List[Dataset]]:
    """
    Create all objects, and populate the axis dataframes.

    Returns: the filtered datasets that will be included. This is simply
    an optimization to allow subsequent X matrix writing to skip unused
    datasets.
    """
    logging.info("Build step 2 - axis creation - started")

    top_level_collection = create_top_level_collections(soma_path)

    # Create axis
    for e in experiment_builders:
        e.create(data_collection=top_level_collection[CENSUS_DATA_NAME])
        assert soma.Experiment(e.se_uri).exists()

    # Write obs axis and accumulate var axis (and remember the datasets that pass our filter)
    filtered_datasets = []
    N = len(datasets) * len(experiment_builders)
    n = 1
    for (dataset, ad) in open_anndata(assets_path, datasets, backed="r"):
        dataset_total_cell_count = 0
        for e in experiment_builders:
            dataset_total_cell_count += e.accumulate_axes(dataset, ad, progress=(n, N))
            n += 1

        dataset.dataset_total_cell_count = dataset_total_cell_count
        if dataset_total_cell_count > 0:
            filtered_datasets.append(dataset)

    # Commit / write var
    for e in experiment_builders:
        e.commit_axis()
        logging.info(f"Experiment {e.name} will contain {e.n_obs} cells from {e.n_datasets} datasets")

    logging.info("Build step 2 - axis creation - finished")
    return top_level_collection, filtered_datasets


def build_step3_create_X_layers(
    assets_path: str,
    filtered_datasets: List[Dataset],
    experiment_builders: List[ExperimentBuilder],
    args: argparse.Namespace,
) -> None:
    """
    Create and populate all X layers
    """
    logging.info("Build step 3 - X layer creation - started")
    # base_path = args.uri

    # Create X layers
    for e in experiment_builders:
        e.create_X_layers(filtered_datasets)
        e.create_joinid_metadata()

    # Process all X data
    populate_X_layers(assets_path, filtered_datasets, experiment_builders, args)

    # tidy up and finish
    for e in experiment_builders:
        e.commit_X(consolidate=args.consolidate)
        e.commit_presence_matrix(filtered_datasets)

    logging.info("Build step 3 - X layer creation - finished")

def add_git_commit_sha(top_level_collection: soma.Collection) -> None:
    sha = get_git_commit_sha()
    top_level_collection.metadata["git_commit_sha"] = sha

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
    # hidden option for testing. Will process only the first 'n' datasets
    build_parser.add_argument("--test-first-n", type=int, help=argparse.SUPPRESS)

    # VALIDATE
    subparsers.add_parser("validate", help="Validate an existing cell census build")

    return parser


if __name__ == "__main__":
    # this is very important to do early, before any use of `concurrent.futures`
    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)

    sys.exit(main())
