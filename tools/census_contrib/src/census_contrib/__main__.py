from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Union

import pyarrow as pa
import tiledbsoma as soma

from .args import Arguments
from .metadata import EmbeddingMetadata, load_metadata, validate_metadata
from .save import consolidate_array, create_obsm_like_array
from .util import EagerIterator, get_logger, soma_context
from .validate import validate_embedding

logger = get_logger()


def main() -> int:
    args = Arguments().parse_args()
    setup_logging(args)

    try:
        metadata_path = args.cwd.joinpath(args.metadata)
        logger.info("Load and validate metadata")
        metadata = validate_metadata(load_metadata(metadata_path))

        embedding_path = args.cwd.joinpath(metadata.id)

        if args.cmd == "validate":
            validate_contrib_embedding(embedding_path, metadata)

        else:  # ingest
            logger.info("Ingesting")
            ingest(args, metadata)

            logger.info("Consolidating")
            consolidate_array(embedding_path)

            logger.info("Validating SOMA array")
            validate_contrib_embedding(embedding_path, metadata)

    except (ValueError, TypeError) as e:
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


def ingest(args: Arguments, metadata: EmbeddingMetadata) -> None:
    save_to = args.cwd.joinpath(metadata.id)
    if save_to.exists():
        args.error("SOMA output path already exists")
    save_to.mkdir(parents=True, exist_ok=True)

    with args.ingestor(args, metadata) as emb_pipe:
        assert emb_pipe.type == pa.float32()

        # Create output object
        domains = emb_pipe.domains
        if domains["i"][0] < 0 or domains["j"][0] < 0:
            args.error("Coordinate values in embedding are negative")
        shape = (domains["i"][1] + 1, domains["j"][1] + 1)
        with create_obsm_like_array(
            save_to.as_posix(), value_range=domains["d"], shape=shape, context=soma_context()
        ) as A:
            logger.debug(f"Array created at {save_to.as_posix()}")
            A.metadata["CxG_contrib_metadata"] = metadata.as_json()
            for block in EagerIterator(emb_pipe):
                if len(block) > 0:
                    logger.debug(f"Writing block length {len(block)}")
                    A.write(block.rename_columns(["soma_dim_0", "soma_dim_1", "soma_data"]))


def validate_contrib_embedding(uri: Union[str, Path], expected_metadata: EmbeddingMetadata) -> None:
    """
    Validate embedding where embedding metadata is encoded in the array.

    Raises upon invalid result
    """
    array_path: str = Path(uri).as_posix()

    with soma.open(array_path, context=soma_context()) as A:
        metadata = json.loads(A.metadata["CxG_contrib_metadata"])

    if expected_metadata.as_dict() != metadata:
        raise ValueError("Expected and actual metadata do not match")

    validate_embedding(array_path, EmbeddingMetadata(**metadata))


if __name__ == "__main__":
    sys.exit(main())
