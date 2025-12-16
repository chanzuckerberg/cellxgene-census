import logging
import os
import pathlib
from datetime import UTC, datetime
from typing import Any, cast

import dask.array as da
import dask.distributed
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
import pyarrow as pa
import tiledbsoma
import tiledbsoma as soma
from anndata.experimental import read_elem_lazy
from dask.delayed import Delayed
from tiledbsoma import (
    Axis,
    CoordinateSpace,
    ScaleTransform,
    Scene,
)

from ..build_state import CensusBuildArgs
from ..util import clamp, cpu_count
from .census_summary import create_census_info_organisms, create_census_summary
from .consolidate import submit_consolidate
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
    CENSUS_POINT_CLOUD_PLATFORM_CONFIG,
    CENSUS_SPATIAL_SEQUENCING_NAME,
    SOMA_TileDB_Context,
)
from .manifest import load_manifest
from .mp import create_dask_client, shutdown_dask_cluster
from .source_assets import stage_source_assets
from .summary_cell_counts import create_census_summary_cell_counts
from .util import get_git_commit_sha
from .validate_soma import validate_consolidation, validate_soma

logger = logging.getLogger(__name__)


def prepare_file_system(args: CensusBuildArgs) -> None:
    """Prepares the file system for the builder run."""
    # Don't clobber an existing census build
    if args.soma_path.exists() or args.h5ads_path.exists():
        raise Exception("Census build path already exists - aborting build")

    # Create top-level build directories
    args.soma_path.mkdir(parents=True, exist_ok=False)
    args.h5ads_path.mkdir(parents=True, exist_ok=False)


def build(args: CensusBuildArgs, *, validate: bool = True) -> int:
    """Build.

    Approximately, build steps are:
    1. Download manifest and copy/stage all source assets
    2. Create top-level collections per experiment
    3. Read (parallel) all H5AD and create axis dataframe (serial). Accumulate overall shape of X
    4. Read (parallel) all H5AD assets again, write X layers. Accumulate X summary stats, presence, etc.
    5. Write axis dataframes and summary information
    6. Consolidate
    7. Validate

    Returns:
    int
        Process completion code, 0 on success, non-zero indicating error,
        suitable for providing to sys.exit()
    """
    experiment_builders = make_experiment_builders()

    prepare_file_system(args)

    n_workers = clamp(cpu_count(), 1, args.config.max_worker_processes)
    with create_dask_client(args, n_workers=n_workers, threads_per_worker=1, memory_limit=0) as client:
        # Step 1 - get all source datasets
        organism_ontology_term_ids = sorted({eb.specification.organism_ontology_term_id for eb in experiment_builders})
        datasets = build_step1_get_source_datasets(args, organism_ontology_term_ids=organism_ontology_term_ids)

        # Step 2 - create root collection, and all child objects, but do not populate any dataframes or matrices
        root_collection = build_step2_create_root_collection(args.soma_path.as_posix(), experiment_builders)

        # Step 3 - populate axes
        filtered_datasets = build_step3_populate_obs_and_var_axes(
            args.h5ads_path.as_posix(), datasets, experiment_builders, args
        )

        # Constraining parallelism is critical at this step, as each worker utilizes (max) ~64GiB+ of memory to
        # process the X array (partitions are large to reduce TileDB fragment count, which reduces consolidation time).
        #
        # TODO: when global order writes are supported, processing of much smaller slices will be
        # possible, and this budget should drop considerably. When that is implemented, n_workers should be
        # be much larger (eg., use default value of #CPUs or some such).
        # https://github.com/single-cell-data/TileDB-SOMA/issues/2054
        MEM_BUDGET = 64 * 1024**3
        n_workers = clamp(int(psutil.virtual_memory().total // MEM_BUDGET), 1, args.config.max_worker_processes)
        logger.info(f"Scaling cluster to {n_workers} workers.")
        client.cluster.scale(n_workers)

        # Step 4 - populate X layers
        build_step4_populate_X_layers(args.h5ads_path.as_posix(), filtered_datasets, experiment_builders, args)

        # Prune datasets that we will not use, and do not want to include in the build
        prune_unused_datasets(args.h5ads_path, datasets, filtered_datasets)

        # Scale the cluster up as we are no longer memory constrained in the following phases
        n_workers = clamp(cpu_count(), 1, args.config.max_worker_processes)
        logger.info(f"Scaling cluster to {n_workers} workers.")
        client.cluster.scale(n=n_workers)

        # Step 4a - add spatial information
        build_step4a_add_spatial(args.h5ads_path.as_posix(), filtered_datasets, experiment_builders, args)

        # Step 5- write out dataset manifest and summary information
        build_step5_save_axis_and_summary_info(
            root_collection, experiment_builders, filtered_datasets, args.config.build_tag
        )

        # Temporary work-around. Can be removed when single-cell-data/TileDB-SOMA#1969 fixed.
        tiledb_soma_1969_work_around(root_collection.uri)

        if args.config.consolidate:
            for f in dask.distributed.as_completed(submit_consolidate(root_collection.uri, pool=client, vacuum=True)):
                assert f.result()
        if validate:
            for f in dask.distributed.as_completed(validate_soma(args, client)):
                assert f.result()
        if args.config.consolidate and validate:
            validate_consolidation(args)
        logger.info("Validation & consolidation complete.")

        shutdown_dask_cluster(client)

    return 0


def prune_unused_datasets(assets_path: pathlib.Path, all_datasets: list[Dataset], used_datasets: list[Dataset]) -> None:
    """Remove any staged H5AD not used to build the SOMA object, ie. those which do not contribute at least one cell to the Census."""
    used_dataset_ids = set(d.dataset_id for d in used_datasets)  # noqa: C401
    unused_datasets = [d for d in all_datasets if d.dataset_id not in used_dataset_ids]
    assert all(d.dataset_total_cell_count == 0 for d in unused_datasets)
    assert all(d.dataset_total_cell_count > 0 for d in used_datasets)
    assert used_dataset_ids.isdisjoint(set(d.dataset_id for d in unused_datasets))  # noqa: C401

    for d in unused_datasets:
        logger.debug(f"Removing unused H5AD {d.dataset_h5ad_path}")
        os.remove(assets_path / d.dataset_h5ad_path)


def populate_root_collection(root_collection: soma.Collection) -> soma.Collection:
    """Create the root SOMA collection for the Census.

    Returns the root collection.
    """
    # Set root metadata for the experiment
    root_collection.metadata["created_on"] = datetime.now(tz=UTC).isoformat(timespec="seconds")

    sha = get_git_commit_sha()
    root_collection.metadata["git_commit_sha"] = sha

    # Create sub-collections for experiments, etc.
    for n in [CENSUS_INFO_NAME, CENSUS_DATA_NAME, CENSUS_SPATIAL_SEQUENCING_NAME]:
        root_collection.add_new_collection(n)

    return root_collection


def build_step1_get_source_datasets(
    args: CensusBuildArgs,
    organism_ontology_term_ids: list[str] | None = None,
) -> list[Dataset]:
    """Stage source H5AD assets locally, either from manifest or by querying CELLxGENE
    REST API.

    organism_ontology_term_ids: if provided, only stage the datasets the API metadata
    associates with these organisms. Otherwise, all datasets with the desired CxG
    schema version will be staged (whether or not we have a SOMAExperimentSpecification
    for their organism). Ignored if a manifest file is provided (since the manifest
    enumerates the datasets to be staged).
    """
    logger.info("Build step 1 - get source assets - started")

    # Load manifest defining the datasets
    all_datasets = load_manifest(
        args.config.manifest,
        args.config.dataset_id_blocklist_uri,
        organism_ontology_term_ids=organism_ontology_term_ids if not args.config.manifest else None,
    )
    if len(all_datasets) == 0:
        logger.error("No H5AD files in the manifest (or we can't find the files)")
        raise RuntimeError("No H5AD files in the manifest (or we can't find the files)")

    # sort encourages (does not guarantee) largest files processed first
    datasets = sorted(all_datasets, key=lambda d: d.asset_h5ad_filesize)

    # Testing/debugging hook - hidden option
    if args.config.test_first_n is not None and abs(args.config.test_first_n) > 0:
        if args.config.test_first_n > 0:
            # Process the N smallest datasets
            datasets = datasets[: args.config.test_first_n]
        else:
            # Process the N largest datasets
            datasets = datasets[args.config.test_first_n :]

    # Stage all files
    stage_source_assets(datasets, args)

    logger.info("Build step 1 - get source assets - finished")
    return datasets


def build_step2_create_root_collection(soma_path: str, experiment_builders: list[ExperimentBuilder]) -> soma.Collection:
    """Create all objects.

    Returns: the root collection.
    """
    logger.info("Build step 2 - Create root collection - started")

    with soma.Collection.create(soma_path, context=SOMA_TileDB_Context()) as root_collection:
        populate_root_collection(root_collection)

        for e in experiment_builders:
            e.create(census_data=root_collection[e.specification.root_collection])

        logger.info("Build step 2 - Create root collection - finished")
        return root_collection


def build_step3_populate_obs_and_var_axes(
    base_path: str,
    datasets: list[Dataset],
    experiment_builders: list[ExperimentBuilder],
    args: CensusBuildArgs,
) -> list[Dataset]:
    """Populate obs and var axes.

    Method:
    1. Concat all obs dataframes
    2. Union all var dataframes
    3. Calculate derived stats and filter datasets by used/not-used
    4. Stash in the ExperimentBuilder

    Accumulation is parallelized; summarization is not parallelized -- it is fast enough
    and benefits from the simplicity.
    """

    def count_cells_per_dataset(
        datasets: list[Dataset],
        accumulated: list[tuple[ExperimentBuilder, tuple[pd.DataFrame, pd.DataFrame]]],
    ) -> None:
        # per dataset, calculate and date total cell count
        cells_per_dataset = (
            pd.concat(obs.value_counts("dataset_id") for _, (obs, _) in accumulated if len(obs))
            .groupby("dataset_id")
            .sum()
        )
        for dataset in datasets:
            dataset.dataset_total_cell_count += cells_per_dataset.get(dataset.dataset_id, default=0)

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
    filtered_datasets: list[Dataset],
    experiment_builders: list[ExperimentBuilder],
    args: CensusBuildArgs,
) -> None:
    """Populate X layers."""
    logger.info("Build step 4 - Populate X layers - started")

    # Process all X data
    for eb in reopen_experiment_builders(experiment_builders):
        eb.create_X_with_layers()

    populate_X_layers(assets_path, filtered_datasets, experiment_builders, args)

    for eb in reopen_experiment_builders(experiment_builders):
        eb.populate_presence_matrix(filtered_datasets)

    logger.info("Build step 4 - Populate X layers - finished")


def build_step4a_add_spatial(
    assets_path: str,
    filtered_datasets: list[Dataset],
    experiment_builders: list[ExperimentBuilder],
    args: CensusBuildArgs,
) -> None:
    """Populate spatial info.

    For each experiment with spatial features:

    1. Iterate through intersection of passed filtered_datasets, and datasets already in this experiment
        1. Collect spatial metadata
        2. Define a delayed task for writing images
        3. Write point clouds
    2. Write images in parallel (they are much larger than other spatial information)
    3. Write spatial presence table
    """
    logger.info("Build step 4a - Populate spatial info - started")
    import h5py
    import pyarrow as pa
    from anndata.io import read_elem

    h5ad_path = args.h5ads_path
    client = dask.distributed.Client.current()

    # Iterate through datasets, extracting fields and setting up delayed tasks
    for eb in reopen_experiment_builders(experiment_builders):
        if not eb.specification.is_exclusively_spatial():
            continue

        logger.debug(f"Writing spatial info to Experiment: {eb.name} at {eb.experiment_uri}")

        datasets = [d for d in filtered_datasets if d.dataset_id in eb.dataset_obs_joinid_start]
        spatial_collection = cast(soma.Experiment, eb.experiment)["spatial"]
        obs_spatial_presences: list[pd.DataFrame] = []  # Dataframes mapping observations to scenes
        write_tasks: list[Delayed] = []  # Image writing tasks to be computed in parallel later

        for d in datasets:
            logger.debug(f"Writing spatial info from {d.dataset_id}")
            scene = spatial_collection.add_new_collection(d.dataset_id, kind=Scene)

            coord_space = CoordinateSpace((Axis(name="y", unit="pixels"), Axis(name="x", unit="pixels")))
            scene.coordinate_space = coord_space

            # Default value. Will be redefined if the dataset is visium (e.g. has a radius defined)
            point_radius = 2.0

            with h5py.File(h5ad_path / d.dataset_h5ad_path) as f:
                # Verify that we're only looking at a single slide dataset
                is_single = read_elem(f["uns/spatial/is_single"])
                assert is_single

                tissue_pos = read_elem(f["obsm/spatial"])
                spatial_group = f["uns/spatial"]

                if len(spatial_group) > 1:
                    # If there is more than one element this dataset has images we should include
                    assert (
                        len(spatial_group) == 2
                    ), f"Found {list(spatial_group)} in {d.dataset_h5ad_path}"  # No image for slide-seqv2
                    _keys = list(spatial_group)
                    # This flag seems wholly uneccesary since you can tell by the number of keys
                    _keys.remove("is_single")
                    library_id = _keys[0]
                    del _keys
                    # There are images
                    spatial_library_info: h5py.Group = spatial_group[library_id]
                    scale_factors = read_elem(spatial_library_info["scalefactors"])
                    point_radius = scale_factors["spot_diameter_fullres"] / 2
                    write_tasks.extend(
                        add_image_collection(
                            scene, library_id, coord_space, spatial_group[library_id], scale_factors=scale_factors
                        )
                    )

            obs = cast(pd.DataFrame, eb.obs_df).query(f"dataset_id == '{d.dataset_id}'")

            # Locations
            logger.debug("Writing locations")

            loc = pd.DataFrame(tissue_pos, columns=["y", "x"])
            loc["soma_joinid"] = obs["soma_joinid"].array
            loc_pa = pa.Table.from_pandas(loc, preserve_index=False)
            scene.add_new_collection("obsl")

            with scene.add_new_point_cloud_dataframe(
                "loc",
                "obsl",
                schema=loc_pa.schema,
                coordinate_space=coord_space,
                transform=tiledbsoma.IdentityTransform(("y", "x"), ("y", "x")),
                domain=[(loc["y"].min(), loc["y"].max()), (loc["x"].min(), loc["x"].max())],
                platform_config=CENSUS_POINT_CLOUD_PLATFORM_CONFIG,
            ) as loc_sink:
                loc_sink.write(loc_pa)

                loc_sink.metadata["soma_geometry"] = point_radius
                loc_sink.metadata["soma_geometry_type"] = "radius"

            obs_spatial_presences.append(obs[["soma_joinid", "dataset_id"]].rename(columns={"dataset_id": "scene_id"}))

        logger.debug("Writing images")
        client.compute(write_tasks, sync=True)

        if len(obs_spatial_presences) == 0:
            logger.warn(f"No scenes found for spatial experiment at {eb.experiment_uri}")
            continue

        logger.debug(f"Creating obs_spatial_presence table for {len(obs_spatial_presences)} scenes")
        obs_spatial_presence = pa.Table.from_pandas(
            pd.concat(obs_spatial_presences, ignore_index=True)
            .reset_index(drop=True)
            # TODO: tiledbsoma currently doesn't let us use a categorical column as an index
            # https://github.com/single-cell-data/TileDB-SOMA/issues/3743
            # .astype({"scene_id": "category"})
            # Add in all True boolean column for interpretation as sparse matrix
            .assign(data=True)
        )

        with cast(soma.Experiment, eb.experiment).add_new_dataframe(
            "obs_spatial_presence",
            schema=obs_spatial_presence.schema,
            index_column_names=["soma_joinid", "scene_id"],
            domain=[
                (np.min(obs_spatial_presence["soma_joinid"]), np.max(obs_spatial_presence["soma_joinid"])),
                ("", ""),
            ],
        ) as df_store:
            df_store.write(obs_spatial_presence)


class TileDBSOMADenseArrayWriteWrapper:
    def __init__(
        self,
        uri: str,
    ):
        """Wrapper around tiledbsoma dense array that providing an interface compatible with dask.array."""
        self.uri = uri

    def __setitem__(self, k: tuple[slice, ...], v: npt.NDArray[Any]) -> None:
        with tiledbsoma.open(self.uri, mode="w") as soma_array:
            soma_array.write(k, pa.Tensor.from_numpy(v))


def write_dask_array_to_existing_tiledbsoma(soma_array: tiledbsoma.DenseNDArray, value: da.Array) -> Delayed:
    wrapped_soma_array = TileDBSOMADenseArrayWriteWrapper(soma_array.uri)

    return value.store(
        wrapped_soma_array,
        lock=False,
        compute=False,
    )


def add_image_collection(
    scene: soma.Collection,
    key: str,
    coordinate_space: CoordinateSpace,
    spatial_library_info: h5py.Group,
    scale_factors: dict[str, float],
) -> list[Delayed]:
    """Creates destination in tiledb store for images for a dataset, returning a delayed task to write the images."""
    image_dict = {k: read_elem_lazy(spatial_library_info["images"][k]) for k in spatial_library_info["images"].keys()}

    scale_transform = ScaleTransform(
        ("y", "x"),
        ("y", "x"),
        (
            scale_factors["tissue_hires_scalef"],
            scale_factors["tissue_hires_scalef"],
        ),
    )

    # Images
    write_tasks = []
    logger.debug(f"Writing images {list(image_dict)}")
    scene.add_new_collection("img")

    hires_image = np.transpose(image_dict["hires"], (2, 0, 1))

    if hires_image.shape[0] == 4:
        # We have an RGBA image, but only want RGB
        hires_image = hires_image[:3]

    multiscale_image = scene.add_new_multiscale_image(
        key=key,
        subcollection="img",
        level_key="hires",
        transform=scale_transform,
        coordinate_space=coordinate_space,
        level_shape=hires_image.shape,
        type=pa.from_numpy_dtype(hires_image.dtype),
    )

    write_tasks.append(write_dask_array_to_existing_tiledbsoma(multiscale_image["hires"], hires_image))

    return write_tasks


def build_step5_save_axis_and_summary_info(
    root_collection: soma.Collection,
    experiment_builders: list[ExperimentBuilder],
    filtered_datasets: list[Dataset],
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
        create_census_info_organisms(census_info, experiment_builders)

    logger.info("Build step 5 - Save axis and summary info - finished")


def tiledb_soma_1969_work_around(census_uri: str) -> None:
    """Remove any inserted bounding box metadata.

    See single-cell-data/TileDB-SOMA#1969 and other issues related.
    """
    bbox_metadata_keys = [
        "soma_dim_0_domain_lower",
        "soma_dim_0_domain_upper",
        "soma_dim_1_domain_lower",
        "soma_dim_1_domain_upper",
    ]

    def _walk_tree(C: soma.Collection) -> list[str]:
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
