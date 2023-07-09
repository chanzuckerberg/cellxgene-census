import argparse
import logging
import pathlib
import sys
from datetime import datetime

from ..build_state import CensusBuildArgs, CensusBuildConfig
from ..util import process_init
from .build_soma import build
from .validate_soma import validate


def main() -> int:
    cli_parser = create_args_parser()
    cli_args = cli_parser.parse_args()
    assert cli_args.subcommand in ["build", "validate"]

    config = CensusBuildConfig(**cli_args.__dict__)
    args = CensusBuildArgs(working_dir=pathlib.PosixPath(cli_args.uri), config=config)
    logging.info(args)
    process_init(args)

    cc = 0
    if cli_args.subcommand == "build":
        cc = build(args)

    if cc == 0 and (cli_args.subcommand == "validate" or cli_args.validate):
        validate(args)

    return cc


def create_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cellxgene_census_builder.build_soma")
    parser.add_argument("uri", type=str, help="Census top-level URI")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity")
    parser.add_argument(
        "-mp",
        "--multi-process",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use multiple processes",
    )
    parser.add_argument("--max-workers", type=int, help="Concurrency")
    parser.add_argument(
        "--build-tag",
        type=str,
        default=datetime.now().astimezone().date().isoformat(),
        help="Census build tag (default: current date is ISO8601 format)",
    )

    subparsers = parser.add_subparsers(required=True, dest="subcommand")

    # BUILD
    build_parser = subparsers.add_parser("build", help="Build the Census")
    build_parser.add_argument(
        "--manifest",
        type=argparse.FileType("r"),
        help="Manifest file",
    )
    build_parser.add_argument(
        "--validate", action=argparse.BooleanOptionalAction, default=True, help="Validate immediately after build"
    )
    build_parser.add_argument(
        "--consolidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Consolidate TileDB objects after build",
    )
    # hidden option for testing by devs. Will process only the first 'n' datasets
    build_parser.add_argument("--test-first-n", type=int)
    # hidden option for testing by devs. Allow for WIP testing by devs.
    build_parser.add_argument("--test-disable-dirty-git-check", action=argparse.BooleanOptionalAction)

    # VALIDATE
    subparsers.add_parser("validate", help="Validate an existing Census build")

    return parser


if __name__ == "__main__":
    sys.exit(main())
