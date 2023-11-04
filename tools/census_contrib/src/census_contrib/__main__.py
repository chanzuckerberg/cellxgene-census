from __future__ import annotations

import logging
import sys

from .args import Arguments
from .metadata import load_metadata, validate_metadata
from .save import save_as_soma


def main() -> int:
    args = Arguments().parse_args()
    setup_logging(args)

    args.logger.info("Starting")
    metadata = load_metadata(args)
    args.logger.info("Validating metadata")
    validate_metadata(args, metadata)
    args.logger.info("Loading embedding")
    embedding = args.ingestor(args, metadata)
    args.logger.info("Saving")
    save_as_soma(args, metadata, embedding)
    args.logger.info("Finished")
    return 0


def setup_logging(args: Arguments) -> None:
    level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose == 1 else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)

    logger = logging.getLogger("census_contrib")
    args.logger = logger


if __name__ == "__main__":
    sys.exit(main())
