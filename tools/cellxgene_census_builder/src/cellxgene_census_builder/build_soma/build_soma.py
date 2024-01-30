import gc
import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import Iterator, List

import tiledbsoma as soma

from ..build_state import CensusBuildArgs
from .anndata import AnnDataProxy, open_anndata
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
    CXG_OBS_COLUMNS_READ,
    CXG_VAR_COLUMNS_READ,
    SOMA_TileDB_Context,
)
from .manifest import load_manifest
from .mp import EagerIterator, create_thread_pool_executor
from .source_assets import stage_source_assets
from .summary_cell_counts import create_census_summary_cell_counts
from .util import get_git_commit_sha, is_git_repo_dirty

logger = logging.getLogger(__name__)


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
    filtered_datasets = build_step3_populate_obs_and_var_axes(
        args.h5ads_path.as_posix(), datasets, experiment_builders, args
    )

    # Prune datasets that we will not use, and do not want to include in the build
    prune_unused_datasets(args.h5ads_path, datasets, filtered_datasets)

    # Step 4 - populate X layers
    build_step4_populate_X_layers(args.h5ads_path.as_posix(), filtered_datasets, experiment_builders, args)
    gc.collect()

    # Step 5- write out dataset manifest and summary information
    build_step5_save_axis_and_summary_info(
        root_collection, experiment_builders, filtered_datasets, args.config.build_tag
    )

    # Step 6 - create and save derived artifacts
    build_step6_save_derived_data(root_collection, experiment_builders, args)

    # Temporary work-around. Can be removed when single-cell-data/TileDB-SOMA#1969 fixed.
    tiledb_soma_1969_work_around(root_collection.uri)

    # consolidate TileDB data
    if args.config.consolidate:
        consolidate(args, root_collection.uri)

    return 0


def prune_unused_datasets(assets_path: pathlib.Path, all_datasets: List[Dataset], used_datasets: List[Dataset]) -> None:
    """Remove any staged H5AD not used to build the SOMA object, ie. those which do not contribute at least one cell to the Census"""
    used_dataset_ids = set(d.dataset_id for d in used_datasets)
    unused_datasets = [d for d in all_datasets if d.dataset_id not in used_dataset_ids]
    assert all(d.dataset_total_cell_count == 0 for d in unused_datasets)
    assert all(d.dataset_total_cell_count > 0 for d in used_datasets)
    assert used_dataset_ids.isdisjoint(set(d.dataset_id for d in unused_datasets))

    for d in unused_datasets:
        logger.debug(f"Removing unused H5AD {d.dataset_h5ad_path}")
        os.remove(assets_path / d.dataset_h5ad_path)


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
    logger.info("Build step 1 - get source assets - started")

    # Load manifest defining the datasets
    all_datasets = load_manifest(args.config.manifest, args.config.dataset_id_blocklist_uri)
    if len(all_datasets) == 0:
        logger.error("No H5AD files in the manifest (or we can't find the files)")
        raise RuntimeError("No H5AD files in the manifest (or we can't find the files)")

    # Testing/debugging hook - hidden option
    if args.config.test_first_n is not None and args.config.test_first_n > 0:
        # Process the N smallest datasets
        datasets = sorted(all_datasets, key=lambda d: d.asset_h5ad_filesize)[0 : args.config.test_first_n]

    else:
        datasets = all_datasets

    # Stage all files
    stage_source_assets(datasets, args)

    logger.info("Build step 1 - get source assets - finished")
    return datasets


def accumulate_axes(
    assets_path: str, datasets: List[Dataset], experiment_builders: List[ExperimentBuilder], args: CensusBuildArgs
) -> List[Dataset]:
    filtered_datasets = []
    N = len(datasets) * len(experiment_builders)
    n = 0

    with create_thread_pool_executor() as pool:
        adata_iter: Iterator[tuple[Dataset, AnnDataProxy]] = (
            (
                dataset,
                open_anndata(
                    assets_path, dataset, obs_column_names=CXG_OBS_COLUMNS_READ, var_column_names=CXG_VAR_COLUMNS_READ
                ),
            )
            for dataset in datasets
        )
        if args.config.multi_process:
            adata_iter = EagerIterator(adata_iter, pool)

        for dataset, ad in adata_iter:
            dataset_total_cell_count = 0
            for eb in experiment_builders:
                n += 1
                logger.info(f"{eb.name}: filtering dataset '{dataset.dataset_id}' ({n} of {N})")
                ad_filtered = eb.filter_anndata_cells(ad)

                if len(ad_filtered.obs) == 0:  # type:ignore
                    logger.info(f"{eb.name} - H5AD has no data after filtering, skipping {dataset.dataset_h5ad_path}")
                    continue

                # accumulate `obs` and `var` data
                dataset_total_cell_count += eb.accumulate_axes(dataset, ad_filtered)

            # dataset passes filter if either experiment includes cells from the dataset
            if dataset_total_cell_count > 0:
                filtered_datasets.append(dataset)
                dataset.dataset_total_cell_count = dataset_total_cell_count

    for eb in experiment_builders:
        eb.finalize_obs_axes()
        logger.info(f"Experiment {eb.name} will contain {eb.n_obs} cells from {eb.n_datasets} datasets")

    return filtered_datasets


def build_step2_create_root_collection(soma_path: str, experiment_builders: List[ExperimentBuilder]) -> soma.Collection:
    """
    Create all objects

    Returns: the root collection.
    """
    logger.info("Build step 2 - Create root collection - started")

    with soma.Collection.create(soma_path, context=SOMA_TileDB_Context()) as root_collection:
        populate_root_collection(root_collection)

        for e in experiment_builders:
            e.create(census_data=root_collection[CENSUS_DATA_NAME])

        logger.info("Build step 2 - Create root collection - finished")
        return root_collection


def build_step3_populate_obs_and_var_axes(
    assets_path: str, datasets: List[Dataset], experiment_builders: List[ExperimentBuilder], args: CensusBuildArgs
) -> List[Dataset]:
    """
    Populate obs and var axes. Filter cells from datasets for each experiment, as obs is built.
    """
    logger.info("Build step 3 - Populate obs and var axes - started")

    filtered_datasets = accumulate_axes(assets_path, datasets, experiment_builders, args)
    logger.info(f"({len(filtered_datasets)} of {len(datasets)}) datasets suitable for processing.")

    for e in experiment_builders:
        e.populate_var_axis()

    assign_dataset_soma_joinids(filtered_datasets)

    logger.info("Build step 3 - Populate obs and var axes - finished")

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
    logger.info("Build step 4 - Populate X layers - started")

    # Process all X data
    for eb in reopen_experiment_builders(experiment_builders):
        eb.create_X_with_layers()

    populate_X_layers(assets_path, filtered_datasets, experiment_builders, args)

    for eb in reopen_experiment_builders(experiment_builders):
        eb.populate_presence_matrix(filtered_datasets)

    logger.info("Build step 4 - Populate X layers - finished")


def build_step5_save_axis_and_summary_info(
    root_collection: soma.Collection,
    experiment_builders: List[ExperimentBuilder],
    filtered_datasets: List[Dataset],
    build_tag: str,
) -> None:
    logger.info("Build step 5 - Save axis and summary info - started")

    for eb in reopen_experiment_builders(experiment_builders):
        eb.write_obs_dataframe()
        eb.write_var_dataframe()

    with soma.Collection.open(root_collection[CENSUS_INFO_NAME].uri, "w", context=SOMA_TileDB_Context()) as census_info:
        create_dataset_manifest(census_info, filtered_datasets)
        create_census_summary_cell_counts(census_info, [e.census_summary_cell_counts for e in experiment_builders])
        create_census_summary(census_info, experiment_builders, build_tag)

    logger.info("Build step 5 - Save axis and summary info - finished")


def build_step6_save_derived_data(
    root_collection: soma.Collection, experiment_builders: List[ExperimentBuilder], args: CensusBuildArgs
) -> None:
    logger.info("Build step 6 - Creating derived objects - started")

    for eb in reopen_experiment_builders(experiment_builders):
        eb.write_X_normalized(args)

        # TODO: to simplify code at some build time expense, we could move
        # feature presence matrix building into this step, and build from
        # X['raw'] rather than building from source H5AD.

    logger.info("Build step 6 - Creating derived objects - finished")
    return


def tiledb_soma_1969_work_around(census_uri: str) -> None:
    """See single-cell-data/TileDB-SOMA#1969 and other issues related. Remove any inserted bounding box metadata"""

    bbox_metadata_keys = [
        "soma_dim_0_domain_lower",
        "soma_dim_0_domain_upper",
        "soma_dim_1_domain_lower",
        "soma_dim_1_domain_upper",
    ]

    def _walk_tree(C: soma.Collection) -> List[str]:
        assert C.soma_type in ["SOMACollection", "SOMAExperiment", "SOMAMeasurement"]
        uris = []
        for soma_obj in C.values():
            type = soma_obj.soma_type
            if type == "SOMASparseNDArray":
                uris.append(soma_obj.uri)
            elif type in ["SOMACollection", "SOMAExperiment", "SOMAMeasurement"]:
                uris += _walk_tree(soma_obj)

        return uris

    with soma.open(census_uri, mode="r") as census:
        sparse_ndarray_uris = _walk_tree(census)

    for uri in sparse_ndarray_uris:
        logger.info(f"tiledb_soma_1969_work_around: deleting bounding box from {uri}")
        with soma.open(uri, mode="w") as A:
            for key in bbox_metadata_keys:
                del A.metadata[key]
