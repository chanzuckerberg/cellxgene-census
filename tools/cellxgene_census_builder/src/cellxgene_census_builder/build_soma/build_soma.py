import logging
import os
import pathlib
from datetime import UTC, datetime
from typing import cast

import dask.array as da
import dask.distributed
import h5py
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import tiledbsoma
import tiledbsoma as soma
from anndata.experimental import read_elem_as_dask

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
    CENSUS_SPATIAL_NAME,
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
        datasets = build_step1_get_source_datasets(args)

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

        client.cluster.scale(16)  # TODO: Come up with some heuristic/ decide on chunk sizes for writing images
        build_step4a_add_spatial(args.h5ads_path.as_posix(), filtered_datasets, experiment_builders, args)

        # Step 5- write out dataset manifest and summary information
        build_step5_save_axis_and_summary_info(
            root_collection, experiment_builders, filtered_datasets, args.config.build_tag
        )

        # Temporary work-around. Can be removed when single-cell-data/TileDB-SOMA#1969 fixed.
        tiledb_soma_1969_work_around(root_collection.uri)

        # Scale the cluster up as we are no longer memory constrained in the following phases
        n_workers = clamp(cpu_count(), 1, args.config.max_worker_processes)
        logger.info(f"Scaling cluster to {n_workers} workers.")
        client.cluster.scale(n=n_workers)

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
    for n in [CENSUS_INFO_NAME, CENSUS_DATA_NAME, CENSUS_SPATIAL_NAME]:
        root_collection.add_new_collection(n)

    return root_collection


def build_step1_get_source_datasets(args: CensusBuildArgs) -> list[Dataset]:
    logger.info("Build step 1 - get source assets - started")

    # Load manifest defining the datasets
    all_datasets = load_manifest(args.config.manifest, args.config.dataset_id_blocklist_uri)
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
            # TODO (spatial): Confirm the decision that we are clearly separating
            # experiments containing spatial assays from experiments not containing
            # spatial assays. That is, an experiment should never contain assays from
            # spatial and non-spatial modalities
            if e.specification.is_exclusively_spatial():
                e.create(census_data=root_collection[CENSUS_SPATIAL_NAME])
            else:
                e.create(census_data=root_collection[CENSUS_DATA_NAME])

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
            dataset.dataset_total_cell_count += cast(
                int,
                cells_per_dataset.get(dataset.dataset_id, default=0),
            )

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
    """Populate spatial info."""
    logger.info("Build step 4a - Populate spatial info - started")
    import h5py
    import pyarrow as pa
    from anndata.experimental import read_elem
    from tiledbsoma import (
        Scene,
    )

    h5ad_path = args.h5ads_path
    client = dask.distributed.Client.current()

    # Add images
    for eb in reopen_experiment_builders(experiment_builders):
        if not eb.specification.is_exclusively_spatial():
            continue

        datasets = [d for d in filtered_datasets if d.dataset_id in eb.dataset_obs_joinid_start]

        # TODO: figure out how to remove this cast
        spatial_collection = cast(soma.Experiment, eb.experiment)["spatial"]
        # Starting with an empty dataframe for the missing case
        # obs_scenes: list[pd.DataFrame] = [
        #     pd.DataFrame({"soma_joinid": np.array([], dtype=np.int64), "scene_id": np.array([], dtype=object)})
        # ]
        obs_scenes: list[pd.DataFrame] = []
        write_tasks = []

        for d in datasets:
            logger.debug(f"Writing spatial info from {d.dataset_id}")
            scene = spatial_collection.add_new_collection(d.dataset_id, kind=Scene)

            with h5py.File(h5ad_path / d.dataset_h5ad_path) as f:
                tissue_pos = read_elem(f["obsm/spatial"])
                is_single = read_elem(f["uns/spatial/is_single"])
                assert is_single
                spatial_group = f["uns/spatial"]
                # spatial_dict = read_elem(f["uns/spatial"])
                if len(spatial_group) > 1:
                    assert (
                        len(spatial_group) == 2
                    ), f"Found {list(spatial_group)} in {d.dataset_h5ad_path}"  # No image for slide-seqv2
                    _keys = list(spatial_group)
                    # This flag seems wholly uneccesary since you can tell by the number of keys
                    _keys.remove("is_single")
                    library_id = _keys[0]
                    del _keys
                    # There are images
                    write_tasks.extend(add_image_collection(scene, spatial_group[library_id]))

            obs = cast(pd.DataFrame, eb.obs_df).query(f"dataset_id == '{d.dataset_id}'")

            # Locations
            logger.debug("Writing locations")

            loc = pd.DataFrame(tissue_pos, columns=["x", "y"])
            loc["soma_joinid"] = obs["soma_joinid"].array
            loc_pa = pa.Table.from_pandas(loc, preserve_index=False)
            obsl = scene.add_new_collection("obsl")
            with obsl.add_new_dataframe(
                "loc",
                schema=loc_pa.schema,
                index_column_names=["soma_joinid"],
            ) as loc_sink:
                loc_sink.write(loc_pa)

            obs_scenes.append(obs[["soma_joinid", "dataset_id"]].rename(columns={"dataset_id": "scene_id"}))

        client.compute(write_tasks, sync=True)

        if len(obs_scenes) == 0:
            logger.warn(f"No scenes found for spatial experiment at {eb.experiment_uri}")
            continue

        logger.debug(f"Creating obs_scene table for {len(obs_scenes)} scenes")
        obs_scene = pa.Table.from_pandas(
            pd.concat(obs_scenes, ignore_index=True).reset_index(drop=True).astype({"scene_id": "category"})
        )

        with cast(soma.Experiment, eb.experiment).add_new_dataframe(
            "obs_scene",
            schema=obs_scene.schema,
        ) as df_store:
            df_store.write(obs_scene)


class TileDBSOMADenseArrayWriteWrapper:
    def __init__(
        self,
        uri: str,
    ):
        self.uri = uri

    def __setitem__(self, k: tuple[slice, ...], v: np.ndarray):
        with tiledbsoma.open(self.uri, mode="w") as soma_array:
            soma_array.write(k, pa.Tensor.from_numpy(v))


def write_dask_array_as_tiledbsoma(collection: tiledbsoma.Collection, key: str, value: da.Array):
    # TODO: Would be nice to set chunking here
    soma_array = collection.add_new_dense_ndarray(key, type=pa.from_numpy_dtype(value.dtype), shape=value.shape)

    wrapped_soma_array = TileDBSOMADenseArrayWriteWrapper(soma_array.uri)

    return value.store(
        wrapped_soma_array,
        lock=False,
        compute=False,
    )


def add_image_collection(
    scene: soma.Collection,
    # spatial_library_info: dict[str, Any],
    spatial_library_info: h5py.Group,
) -> None:
    from anndata.experimental import read_elem
    from somacore import (
        Axis,
        CoordinateSpace,
        ScaleTransform,
    )

    img_collection = scene.add_new_collection("img")
    scale_factors = read_elem(spatial_library_info["scalefactors"])
    image_dict = {
        k: read_elem_as_dask(spatial_library_info["images"][k]) for k in spatial_library_info["images"].keys()
    }

    # Coordinate systems

    # pixels_per_spot_radius = 0.5 * scale_factors["spot_diameter_fullres"]
    fullres_to_coords_scale = 65 / scale_factors["spot_diameter_fullres"]

    # Create axes and transformations
    coordinate_system = CoordinateSpace(
        (
            Axis(name="y", units="micrometer"),
            Axis(name="x", units="micrometer"),
        )
    )

    spots_to_coords = ScaleTransform((fullres_to_coords_scale, fullres_to_coords_scale))

    scene.metadata["soma_scene_coordinates"] = coordinate_system.to_json()
    img_metadata = {"soma_scene_coords": spots_to_coords.to_json()}
    # axes_metadata = [
    #     {"name": "c", "type": "channel"},
    #     {"name": "y", "type": "space", "unit": "micrometer"},
    #     {"name": "x", "type": "space", "unit": "micrometer"},
    # ]

    # Images
    write_tasks = []
    logger.debug(f"Writing images {list(image_dict)}")
    for k, v in image_dict.items():
        if k == "fullres":
            scale = (1.0, 1.0, 1.0)
        elif k == "hires":
            scale = (1.0, scale_factors["tissue_hires_scalef"], scale_factors["tissue_hires_scalef"])
        elif k == "lowres":
            scale = (1.0, scale_factors["tissue_lowres_scalef"], scale_factors["tissue_lowres_scalef"])
        else:
            raise ValueError(f"Unexpected image name: {k}")

        img_metadata[f"soma_asset_transform_{k}"] = ScaleTransform(scale).to_json()
        write_tasks.append(write_dask_array_as_tiledbsoma(img_collection, k, np.transpose(v, (2, 0, 1))))
        # image_array = DenseNDArray.create(
        #     image_uri,
        #     type=pa.from_numpy_dtype(im.dtype),
        #     shape=im.shape,
        #     # platform_config=platform_config,
        #     # context=context,
        # )
        # tensor = pa.Tensor.from_numpy(im)
        # image_array.write(
        #     (slice(None), slice(None), slice(None)),
        #     tensor,
        # )
        # img_collection.set(k, image_array, use_relative_uri=True)
    img_collection.metadata.update(img_metadata)
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
