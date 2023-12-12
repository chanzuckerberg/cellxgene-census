from __future__ import annotations

import logging
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import scanpy as sc
import scipy.sparse as sp
import tiledbsoma as soma

from .args import Arguments
from .census_util import get_obs_soma_joinids, open_census
from .config import Config
from .metadata import EmbeddingMetadata, load_metadata, validate_metadata
from .save import consolidate_array, consolidate_group, create_obsm_like_array, reduce_float_precision
from .util import (
    EagerIterator,
    blocksize,
    blockwise_axis0_scipy_csr,
    get_logger,
    has_blockwise_iterator,
    soma_context,
    uri_to_path,
)
from .validate import validate_compatible_tiledb_storage_format, validate_embedding

logger = get_logger()


def main() -> int:
    args = Arguments().parse_args()
    setup_logging(args)

    try:
        metadata_path = args.cwd.joinpath(args.metadata)
        logger.info("Load and validate metadata")
        metadata = validate_metadata(args, load_metadata(metadata_path))
        embedding_path = args.cwd.joinpath(metadata.id)

        config = Config(args=args, metadata=metadata)

        if args.cmd == "validate":
            validate_cmd(config, embedding_path)
        elif args.cmd == "qcplots":
            qcplots_cmd(config, embedding_path)
        elif args.cmd.startswith("ingest-"):  # ingest
            ingest_cmd(config, embedding_path)
        elif args.cmd == "inject":
            inject_cmd(config, embedding_path)
        else:
            args.print_help()

    except (ValueError, TypeError) as e:
        if args.verbose:
            traceback.print_exc()

        args.error(str(e))

    logger.info("Finished")
    return 0


def setup_logging(args: Arguments) -> None:
    level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose == 1 else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)

    # turn down some other stuff
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def validate_cmd(config: Config, embedding_path: Path) -> None:
    logger.info("Validating SOMA array")
    validate_contrib_embedding(
        embedding_path, config, skip_storage_version_check=config.args.skip_storage_version_check
    )

    logger.info("Creating QC umaps")
    create_qc_plots(config, embedding_path)


def qcplots_cmd(config: Config, embedding_path: Path) -> None:
    logger.info("Creating QC umaps")
    create_qc_plots(config, embedding_path)


def ingest_cmd(config: Config, embedding_path: Path) -> None:
    args = config.args

    logger.info("Ingesting")
    ingest(config)

    logger.info("Consolidating")
    consolidate_array(embedding_path)

    logger.info("Validating SOMA array")
    validate_contrib_embedding(embedding_path, config, skip_storage_version_check=args.skip_storage_version_check)

    logger.info("Creating QC umaps")
    create_qc_plots(config, embedding_path)


def inject_cmd(config: Config, embedding_path: Path) -> None:
    logger.info("Adding embedding to Census build")
    inject_embedding_into_census_build(config, embedding_path)


def ingest(config: Config) -> None:
    metadata = config.metadata
    args = config.args

    save_to = args.cwd.joinpath(metadata.id)
    if save_to.exists():
        args.error("SOMA output path already exists")
    save_to.mkdir(parents=True, exist_ok=True)

    obs_joinids, obs_shape = get_obs_soma_joinids(config)
    assert obs_joinids[0] == 0 and obs_joinids[-1] == (obs_shape[0] - 1), "Census coordinates are unexpected"

    with args.ingestor(config) as emb_pipe:
        assert emb_pipe.type == pa.float32()

        # Create output object
        domains = emb_pipe.domains
        if domains["i"][0] < 0 or domains["j"][0] < 0:
            args.error("Coordinate values in embedding are negative")
        if domains["i"][1] > obs_shape[0]:
            args.error("Coordinate values in embedding dim 0 are outside obs joinid range")
        if domains["j"] != (0, metadata.n_features - 1):
            args.error("Coordinate values in embedding dim 1 are not (0, n_features)")

        if domains["i"][0] > 0 or domains["i"][1] < (obs_shape[0] - 1):
            logger.warning("Embedding is a subset of cells.")

        assert 0 == domains["j"][0] and (metadata.n_features - 1) == domains["j"][1]
        assert domains["i"][0] <= domains["i"][1] < obs_shape[0]
        shape = (obs_shape[0], metadata.n_features)

        with create_obsm_like_array(
            save_to.as_posix(),
            value_range=domains["d"],
            shape=shape,
            context=soma_context({"sm.check_coord_dups": True}),
        ) as A:
            logger.debug(f"Array created at {save_to.as_posix()}")
            A.metadata["CxG_embedding_info"] = metadata.to_json()
            for block in EagerIterator(emb_pipe):
                assert isinstance(block, pa.Table), "Embedding pipe did not yield an Arrow Table"
                assert block.column_names == ["i", "j", "d"]  # we care about the order
                if len(block) > 0:
                    logger.debug(f"Writing block length {len(block)}")
                    block = reduce_float_precision(block, args.float_precision)
                    A.write(block.rename_columns(["soma_dim_0", "soma_dim_1", "soma_data"]))


def validate_contrib_embedding(uri: Union[str, Path], config: Config, skip_storage_version_check: bool = False) -> None:
    """
    Validate embedding where embedding metadata is encoded in the array.

    Raises upon invalid result
    """
    array_path: str = Path(uri).as_posix()

    with soma.open(array_path, context=soma_context()) as A:
        metadata = EmbeddingMetadata.from_json(A.metadata["CxG_embedding_info"])

    if config.metadata != metadata:
        raise ValueError("Expected and actual metadata do not match")

    if not skip_storage_version_check:
        validate_compatible_tiledb_storage_format(array_path, config)

    validate_embedding(config, array_path)


def load_qc_anndata(
    config: Config,
    embedding: Path,
    obs_value_filter: str,
    obs_columns: List[str],
    emb_name: str,
) -> Optional[sc.AnnData]:
    """Returns None if the value filter excludes all cells"""
    if "soma_joinid" not in obs_columns:
        obs_columns = ["soma_joinid"] + obs_columns

    metadata = config.metadata

    with open_census(census_version=config.metadata.census_version, census_uri=config.args.census_uri) as census:
        experiment = census["census_data"][metadata.experiment_name]
        with experiment.axis_query(
            measurement_name=metadata.measurement_name,
            obs_query=soma.AxisQuery(value_filter=obs_value_filter),
            var_query=soma.AxisQuery(coords=(slice(0, 1),)),  # we don't use X data, so minimize load memory & time
        ) as query:
            if not query.n_obs:
                return None

            # Load AnnData and X[raw]
            adata = query.to_anndata(X_name="raw", column_names={"obs": obs_columns})
            obs_joinids = adata.obs.soma_joinid.to_numpy()

        # Load embedding associated with the obs joinids
        with soma.open(embedding.as_posix(), context=soma_context()) as E:
            # read embedding and obs joinids
            size = blocksize(E.shape[1])
            embeddings = sp.vstack(
                [
                    blk
                    for blk, _ in (
                        E.read(coords=(obs_joinids,)).blockwise(axis=0, size=size, reindex_disable_on_axis=1).scipy()
                        if has_blockwise_iterator()
                        else blockwise_axis0_scipy_csr(E, coords=(obs_joinids,), size=size, reindex_disable_on_axis=1)
                    )
                ]
            )

            embedding_presence_mask = embeddings.getnnz(axis=1) != 0
            embeddings = embeddings[embedding_presence_mask, :].toarray()
            adata = adata[embedding_presence_mask, :]

            # Save as an obsm layer
            adata.obsm[emb_name] = embeddings

    return adata


def create_qc_plots(config: Config, embedding: Path) -> None:
    sc._settings.settings.autoshow = False
    sc._settings.settings.figdir = (config.args.cwd / "figures").as_posix()

    def make_random_palette(n_colors: int) -> List[str]:
        rng = np.random.default_rng()
        colors = rng.integers(0, 0xFFFFFF, size=n_colors, dtype=np.uint32)
        return [f"#{c:06X}" for c in colors]

    cases = {
        "spleen": "is_primary_data == True and tissue_general == 'spleen'",
        "spinal_cord": "is_primary_data == True and tissue_general == 'spinal cord'",
    }
    emb_name = "X_emb"

    color_by_columns = ["cell_type", "dataset_id", "assay"]

    for k, filter in cases.items():
        logger.info(f"Loading AnnData for QC UMAP for {k}")
        adata = load_qc_anndata(config, embedding, filter, color_by_columns, emb_name)
        if not adata:
            logger.info(f"Zero cells available for {k}, skipping UMAP")
            continue

        logger.info(repr(adata))

        logger.info(f"Computing neighbor graph for {k}")
        sc.pp.neighbors(adata, use_rep=emb_name)

        logger.info(f"Computing UMAP for {k}")
        sc.tl.umap(adata)

        logger.info(f"Saving UMAP plots for {k}")
        for color_by in color_by_columns:
            n_categories = len(adata.obs[color_by].astype(str).astype("category").cat.categories)
            plot_color_kwargs: Dict[str, Any] = dict(color=color_by)
            # scanpy does a good job until category counts > 102
            if n_categories > len(sc.plotting.palettes.default_102):
                plot_color_kwargs["palette"] = make_random_palette(n_categories)
            sc.pl.umap(adata, title=f"{k}, colored by {color_by}", save=f"_{k}_{color_by}.png", **plot_color_kwargs)


def inject_embedding_into_census_build(config: Config, embedding_src_path: Path) -> None:
    """
    Inject an existing embedding (ingested via this tool) into its corresponding Census build.
    Presumed workflow:
        * build census
        * create embedding(s)
        * inject embeddings
        * publish census

    Assumptions:
    1. The embedding matches the Census build (e.g., soma_joinids)
    2. The Census build and embedding are on the local file system

    Process:
    1. create obsm if needed
    2. copy embedding into obsm dir
    3. clean up embedding metadata
    4. add embedding to obsm group
    5. conslidate obsm
    6. validate that group edits worked, etc.
    """
    metadata = config.metadata
    census_write_path = config.args.census_write_path
    obsm_key = config.args.obsm_key
    obsm_layer_name = obsm_key or metadata.id

    # Pre-checks
    logger.info(f"Injecting embedding into {census_write_path.as_posix()} with name obsm['{obsm_layer_name}']")
    if not census_write_path.is_dir() or not soma.Collection.exists(census_write_path.as_posix()):
        raise ValueError("Census path does not exist, is not on local file system or is not a Collection")
    if not embedding_src_path.is_dir() or not soma.SparseNDArray.exists(embedding_src_path.as_posix()):
        raise ValueError("Embedding path does not exist, is not on local file system or is not a SparseNDArray")

    # Get ms URI, and do additional precautionary checks
    with soma.open(census_write_path.as_posix(), context=soma_context()) as census:
        exp = census["census_data"][metadata.experiment_name]
        if metadata.n_embeddings > exp.obs.count:
            raise ValueError("Metadata and census obs shape mismatch")
        with soma.open(embedding_src_path.as_posix(), context=soma_context()) as E:
            if E.shape[0] != exp.obs.count:
                raise ValueError("Census obs and embedding shape mismatch")
        ms_path = uri_to_path(exp.ms[metadata.measurement_name].uri)

    # Create obsm if it does not already exist
    with soma.open(ms_path.as_posix(), mode="w", context=soma_context()) as ms:
        if "obsm" not in ms:
            logger.info(f"obsm does not exist, adding to {ms_path}")
            ms.add_new_collection("obsm")

    obsm_path = ms_path / "obsm"
    assert soma.Measurement.exists(ms_path.as_posix())
    assert soma.Collection.exists(obsm_path.as_posix())

    # Copy the pre-existing embedding into the obsm collection directory. Raises if exists.
    embedding_dst_path = obsm_path / obsm_layer_name
    logger.info(f"Copy {embedding_src_path.as_posix()} -> {embedding_dst_path.as_posix()}")
    shutil.copytree(embedding_src_path, embedding_dst_path)

    # Add the embedding to the `obsm` collection
    with soma.open(ms_path.as_posix(), mode="w", context=soma_context()) as ms:
        with soma.open(embedding_dst_path.as_posix()) as emb_copy:
            ms["obsm"].set(obsm_layer_name, emb_copy, use_relative_uri=True)

    # consolidate/vacuum all
    consolidate_group(ms_path)
    consolidate_group(obsm_path)
    consolidate_array(embedding_dst_path)


if __name__ == "__main__":
    sys.exit(main())
