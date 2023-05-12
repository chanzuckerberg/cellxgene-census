import gc
import logging
from datetime import datetime, timezone
from typing import List

import tiledbsoma as soma

from ..build_state import CensusBuildArgs
from .anndata import open_anndata
from .census_summary import create_census_summary
from .consolidate import consolidate
from .datasets import Dataset, assign_dataset_soma_joinids, create_dataset_manifest
from .experiment_builder import (
    ExperimentBuilder,
    populate_X_layers,
    reopen_experiment_builders,
)
from .experiment_specs import make_experiment_builders
from .globals import (
    CENSUS_DATA_NAME,
    CENSUS_INFO_NAME,
    SOMA_TileDB_Context,
)
from .manifest import load_manifest
from .source_assets import stage_source_assets
from .summary_cell_counts import create_census_summary_cell_counts
from .util import get_git_commit_sha, is_git_repo_dirty


def prepare_file_system(args: CensusBuildArgs) -> None:
    """
    Prepares the file system for the builder run
    """
    # Don't clobber an existing census build
    if args.soma_path.exists() or args.h5ads_path.exists():
        raise Exception("Census build path already exists - aborting build")

    # Ensure that the git tree is clean
    if not args.config.disable_dirty_git_check and is_git_repo_dirty():
        raise Exception("The git repo has uncommitted changes - aborting build")

    # Create top-level build directories
    args.soma_path.mkdir(parents=True, exist_ok=False)
    args.h5ads_path.mkdir(parents=True, exist_ok=False)


def build(args: CensusBuildArgs) -> int:
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

    experiment_builders = make_experiment_builders()

    prepare_file_system(args)

    # Step 1 - get all source datasets
    datasets = build_step1_get_source_datasets(args)

    # Step 2 - create root collection, and all child objects, but do not populate any dataframes or matrices
    root_collection = build_step2_create_root_collection(args.soma_path.as_posix(), experiment_builders)
    gc.collect()

    # Step 3 - populate axes
    filtered_datasets = build_step3_populate_obs_and_var_axes(args.h5ads_path.as_posix(), datasets, experiment_builders)

    # Step 4 - populate X layers
    build_step4_populate_X_layers(args.h5ads_path.as_posix(), filtered_datasets, experiment_builders, args)
    gc.collect()

    # Step 5- write out dataset manifest and summary information
    build_step5_populate_summary_info(root_collection, experiment_builders, filtered_datasets, args.config.build_tag)

    # consolidate TileDB data
    if args.config.consolidate:
        consolidate(args, root_collection.uri)

    return 0


def populate_root_collection(root_collection: soma.Collection) -> soma.Collection:
    """
    Create the root SOMA collection for the Census.

    Returns the root collection.
    """

    # Set root metadata for the experiment
    root_collection.metadata["created_on"] = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    sha = get_git_commit_sha()
    root_collection.metadata["git_commit_sha"] = sha

    # Create sub-collections for experiments, etc.
    for n in [CENSUS_INFO_NAME, CENSUS_DATA_NAME]:
        root_collection.add_new_collection(n)

    return root_collection


def build_step1_get_source_datasets(args: CensusBuildArgs) -> List[Dataset]:
    logging.info("Build step 1 - get source assets - started")

    # Load manifest defining the datasets
    datasets = load_manifest(args.config.manifest)
    if len(datasets) == 0:
        logging.error("No H5AD files in the manifest (or we can't find the files)")
        raise AssertionError("No H5AD files in the manifest (or we can't find the files)")

    # Testing/debugging hook - hidden option
    if args.config.test_first_n is not None and args.config.test_first_n > 0:
        # Process the N smallest datasets
        datasets = sorted(datasets, key=lambda d: d.asset_h5ad_filesize)[0 : args.config.test_first_n]

    # Stage all files
    stage_source_assets(datasets, args)

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

            if len(ad_filtered.obs) == 0:  # type:ignore
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


def populate_var_axis_and_presence(experiment_builders: List[ExperimentBuilder]) -> None:
    for eb in reopen_experiment_builders(experiment_builders):
        # populate `var`; create empty `presence` now that we have its dimensions
        eb.populate_var_axis()


def build_step2_create_root_collection(soma_path: str, experiment_builders: List[ExperimentBuilder]) -> soma.Collection:
    """
    Create all objects

    Returns: the root collection.
    """
    logging.info("Build step 2 - Create root collection - started")

    with soma.Collection.create(soma_path, context=SOMA_TileDB_Context()) as root_collection:
        populate_root_collection(root_collection)

        for e in experiment_builders:
            e.create(census_data=root_collection[CENSUS_DATA_NAME])

        logging.info("Build step 2 - Create root collection - finished")
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
    args: CensusBuildArgs,
) -> None:
    """
    Populate X layers.
    """
    logging.info("Build step 4 - Populate X layers - started")

    # Process all X data
    for eb in reopen_experiment_builders(experiment_builders):
        eb.create_X_with_layers()

    populate_X_layers(assets_path, filtered_datasets, experiment_builders, args)

    for eb in reopen_experiment_builders(experiment_builders):
        eb.populate_presence_matrix(filtered_datasets)

    logging.info("Build step 4 - Populate X layers - finished")


def build_step5_populate_summary_info(
    root_collection: soma.Collection,
    experiment_builders: List[ExperimentBuilder],
    filtered_datasets: List[Dataset],
    build_tag: str,
) -> None:
    logging.info("Build step 5 - Populate summary info - started")

    with soma.Collection.open(root_collection[CENSUS_INFO_NAME].uri, "w", context=SOMA_TileDB_Context()) as census_info:
        create_dataset_manifest(census_info, filtered_datasets)
        create_census_summary_cell_counts(census_info, [e.census_summary_cell_counts for e in experiment_builders])
        create_census_summary(census_info, experiment_builders, build_tag)

    logging.info("Build step 5 - Populate summary info - finished")
