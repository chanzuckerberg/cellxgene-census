import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import List, cast

import dask
import pandas as pd
import tiledbsoma as soma

from ..build_state import CensusBuildArgs
from ..util import cpu_count
from .census_summary import create_census_summary
from .consolidate import consolidate, start_async_consolidation, stop_async_consolidation, submit_consolidate
from .datasets import Dataset, assign_dataset_soma_joinids, create_dataset_manifest
from .experiment_builder import (
    ExperimentBuilder,
    accumulate_axes_dataframes,
    populate_X_layers,
    post_acc_axes_processing,
    reopen_experiment_builders,
)
from .experiment_specs import make_experiment_builders
from .globals import (
    CENSUS_DATA_NAME,
    CENSUS_INFO_NAME,
    SOMA_TileDB_Context,
)
from .manifest import load_manifest
from .mp import create_dask_client
from .source_assets import stage_source_assets
from .summary_cell_counts import create_census_summary_cell_counts
from .util import get_git_commit_sha, is_git_repo_dirty
from .validate_soma import validate_consolidation, validate_soma

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


def build(args: CensusBuildArgs, *, validate: bool = True) -> int:
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

    with create_dask_client(args, n_workers=cpu_count(), threads_per_worker=2):
        # Step 1 - get all source datasets
        datasets = build_step1_get_source_datasets(args)

        # Step 2 - create root collection, and all child objects, but do not populate any dataframes or matrices
        root_collection = build_step2_create_root_collection(args.soma_path.as_posix(), experiment_builders)

        # Step 3 - populate axes
        filtered_datasets = build_step3_populate_obs_and_var_axes(
            args.h5ads_path.as_posix(), datasets, experiment_builders, args
        )

    # Constraining parallelism is critical at this step, as each worker utilizes 64GiB+ of buffer and will
    # create ~ncores threads during the write phase (this code assumes hosts with 8GiB/core).
    n_workers = max(1, cpu_count() // 16) + 2
    with create_dask_client(args, n_workers=n_workers, threads_per_worker=1, memory_limit=0) as dask_client:
        try:
            if args.config.consolidate:
                consolidator = start_async_consolidation(dask_client, root_collection.uri)

            # Step 4 - populate X layers
            build_step4_populate_X_layers(args.h5ads_path.as_posix(), filtered_datasets, experiment_builders, args)

            # Prune datasets that we will not use, and do not want to include in the build
            prune_unused_datasets(args.h5ads_path, datasets, filtered_datasets)

            # Step 5- write out dataset manifest and summary information
            build_step5_save_axis_and_summary_info(
                root_collection, experiment_builders, filtered_datasets, args.config.build_tag
            )

        finally:
            if consolidator:
                stop_async_consolidation(consolidator)  # blocks until any running consolidate finishes
                del consolidator

        # Temporary work-around. Can be removed when single-cell-data/TileDB-SOMA#1969 fixed.
        tiledb_soma_1969_work_around(root_collection.uri)

        # XXX indented while commented
        # with create_dask_client(args, n_workers=cpu_count(), threads_per_worker=1, memory_limit=None) as dask_client:

        # XXX TODO: scale cluster up once we move validation into Dask

        # Validation and consolidation are done in parallel, thanks to the TileDB
        # concurrency model. The only important constraint is that vacuuming _MUST_
        # be done _after_ validation has completed, to avoid races between readers
        # and deletion.
        valcon_futures: List[dask.distributed.Futures] = []
        if args.config.consolidate:
            valcon_futures += submit_consolidate(root_collection.uri, pool=dask_client, vacuum=False)
        if validate:
            # valcon_futures.append(dask_client.submit(validate_soma, args))
            # until validate is converted to dask, it creates MP pools internally
            # so call directly
            validate_soma(args)
        if valcon_futures:
            dask_client.gather(valcon_futures)

        # Once validated and consolidated, so a second pass with vacuuming, and
        # then validate it worked correctly
        if args.config.consolidate:
            consolidate(args, root_collection.uri)
            validate_consolidation(args)

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
    base_path: str,
    datasets: List[Dataset],
    experiment_builders: List[ExperimentBuilder],
    args: CensusBuildArgs,
) -> List[Dataset]:
    """
    Method:
    1. Concat all obs dataframes
    2. Union all var dataframes
    3. Calculate derived stats and filter datasets by used/not-used
    4. Stash in the ExperimentBuilder

    Accumulation is parallelized; summarization is not parallelized -- it is fast enough
    and benefits from the simplicity.
    """

    def count_cells_per_dataset(
        datasets: List[Dataset],
        accumulated: List[tuple[ExperimentBuilder, tuple[pd.DataFrame, pd.DataFrame]]],
    ) -> None:
        # per dataset, calculate and date total cell count
        cells_per_dataset = (
            pd.concat(obs.value_counts("dataset_id") for _, (obs, _) in accumulated if len(obs))
            .groupby("dataset_id")
            .sum()
        )
        for dataset in datasets:
            dataset.dataset_total_cell_count += cast(int, cells_per_dataset.get(dataset.dataset_id, default=0))  # type: ignore[arg-type]

    logger.info("Build step 3 - accumulate obs and var axes - started")

    accumulated = accumulate_axes_dataframes(base_path, datasets, experiment_builders)

    logger.info("Build step 3 - axis accumulation complete")

    # Determine which datasets are utilized
    count_cells_per_dataset(datasets, accumulated)
    datasets_utilized = list(filter(lambda d: d.dataset_total_cell_count > 0, datasets))
    assign_dataset_soma_joinids(datasets_utilized)

    # summarize axes, add additional columns, etc - saving result into experiment_builders
    post_acc_axes_processing(accumulated)

    logger.info("Build step 3 - accumulate obs and var axes - finished")
    return datasets_utilized


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
                if key in A.metadata:
                    del A.metadata[key]
