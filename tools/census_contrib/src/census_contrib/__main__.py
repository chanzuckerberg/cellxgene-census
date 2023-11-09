from __future__ import annotations

import json
import logging
import pathlib
import sys

import pyarrow as pa
import tiledbsoma as soma

from .args import Arguments
from .load import EmbeddingIJD
from .metadata import EmbeddingMetadata, load_metadata, validate_metadata
from .save import consolidate_array, create_obsm_like_array
from .util import get_logger, soma_context
from .validate import validate_embedding, validate_embeddingIJD

logger = get_logger()


def main() -> int:
    args = Arguments().parse_args()
    setup_logging(args)

    try:
        if args.cmd == "validate":
            expected_metadata = load_metadata(args.metadata)
            validate_contrib_embedding(args.uri, args.accession, expected_metadata)

        else:  # ingestor
            logger.info("Starting")
            metadata = load_metadata(args.metadata)

            logger.info("Validate metadata")
            validate_metadata(metadata)

            logger.info("Loading embedding")
            embedding = args.ingestor(args, metadata)

            logger.info(f"Pre-validate embedding [shape={embedding.shape}]")
            validate_embeddingIJD(embedding, metadata)

            logger.info("Saving")
            save_as_soma(args, metadata, embedding)

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


def save_as_soma(args: Arguments, metadata: EmbeddingMetadata, emb: EmbeddingIJD) -> None:
    save_to = pathlib.PosixPath(args.save_soma_to)
    if save_to.exists():
        args.error("SOMA output path already exists")

    save_to.mkdir(parents=True, exist_ok=True)

    emb_tbl = pa.Table.from_pydict(
        {
            "soma_dim_0": pa.array(emb.i),
            "soma_dim_1": pa.array(emb.j),
            "soma_data": pa.array(emb.d),
        }
    )

    logger.info("Creating SOMA array")
    uri = save_to.as_posix()
    value_range = (emb.d.min(), emb.d.max())
    with create_obsm_like_array(uri, value_range=value_range, shape=emb.shape, context=soma_context()) as A:
        logger.info("Writing SOMA array - starting")
        A.metadata["CxG_accession_id"] = args.accession
        A.metadata["CxG_contrib_metadata"] = metadata.as_json()

        logger.info("Start write")
        A.write(emb_tbl)
        logger.info("Writing SOMA array - completed")


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
