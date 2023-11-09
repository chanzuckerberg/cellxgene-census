from __future__ import annotations

import json
import logging
import pathlib
import sys

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
        if args.cmd == "validate":
            expected_metadata = validate_metadata(load_metadata(args.metadata))
            validate_contrib_embedding(args.uri, args.accession, expected_metadata)

        else:  # ingestor
            logger.info("Load and validate metadata")
            metadata = validate_metadata(load_metadata(args.metadata))

            logger.info("Ingesting")
            ingest(args, metadata)

            logger.info("Consolidating")
            consolidate_array(args.save_soma_to)

            logger.info("Validating SOMA array")
            validate_contrib_embedding(args.save_soma_to, args.accession, metadata)

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
    # TODO
    # 1. validate pipe info (e.g., type, shape, etc) - see validate_embeddingIJD for ideas
    # 2. validate each block

    save_to = pathlib.PosixPath(args.save_soma_to)
    if save_to.exists():
        args.error("SOMA output path already exists")
    save_to.mkdir(parents=True, exist_ok=True)

    with args.ingestor(args, metadata) as emb_pipe:
        assert emb_pipe.type == pa.float32()

        # Create output object
        value_range = emb_pipe.value_range
        with create_obsm_like_array(
            save_to.as_posix(), value_range=value_range, shape=emb_pipe.shape, context=soma_context()
        ) as A:
            logger.debug(f"Array created at {save_to.as_posix()}")
            A.metadata["CxG_accession_id"] = args.accession
            A.metadata["CxG_contrib_metadata"] = metadata.as_json()
            for block in EagerIterator(emb_pipe):
                logger.debug(f"Writing block length {len(block)}")
                A.write(block.rename_columns(["soma_dim_0", "soma_dim_1", "soma_data"]))


def validate_contrib_embedding(uri: str, expected_accession: str, expected_metadata: EmbeddingMetadata) -> None:
    """
    Validate embedding where embedding metadata is encoded in the array.

    Raises upon invalid result
    """
    with soma.open(uri, context=soma_context()) as A:
        accession = A.metadata["CxG_accession_id"]
        metadata = json.loads(A.metadata["CxG_contrib_metadata"])

    if accession != expected_accession:
        raise ValueError("Expected and actual accession do not match")

    if expected_metadata.as_dict() != metadata:
        raise ValueError("Expected and actual metadata do not match")

    validate_embedding(uri, EmbeddingMetadata(**metadata))


if __name__ == "__main__":
    sys.exit(main())
