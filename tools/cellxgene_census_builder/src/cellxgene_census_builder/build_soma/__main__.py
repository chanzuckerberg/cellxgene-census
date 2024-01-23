import argparse
import logging
import pathlib
import sys
from datetime import datetime

import attrs

from ..build_state import CensusBuildArgs, CensusBuildConfig
from ..util import log_process_resource_status, process_init, start_resource_logger
from .build_soma import build
from .validate_soma import validate

logger = logging.getLogger(__name__)


def main() -> int:
    cli_parser = create_args_parser()
    cli_args = cli_parser.parse_args()
    assert cli_args.subcommand in ["build", "validate"]

    # Pass params from CLI arguments _only_ if they exist in the CensusBuildConfig namespace
    default_config = CensusBuildConfig(
        **{
            k: cli_args.__dict__[k]
            for k in cli_args.__dict__.keys() & {f.alias for f in attrs.fields(CensusBuildConfig)}
        }
    )
    args = CensusBuildArgs(working_dir=pathlib.PosixPath(cli_args.working_dir), config=default_config)
    process_init(args)
    logger.info(args)

    start_resource_logger()

    cc = 0
    if cli_args.subcommand == "build":
        cc = build(args)

    if cc == 0 and (cli_args.subcommand == "validate" or cli_args.validate):
        validate(args)

    log_process_resource_status(level=logging.INFO)
    return cc


def create_args_parser() -> argparse.ArgumentParser:
    default_config = CensusBuildConfig()
    parser = argparse.ArgumentParser(prog="cellxgene_census_builder.build_soma")
    parser.add_argument("working_dir", type=str, help="Census build working directory")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity")
    parser.add_argument(
        "-mp",
        "--multi-process",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use multiple processes",
    )
    parser.add_argument(
        "--max_worker_processes",
        type=int,
        default=default_config.max_worker_processes,
        help="Limit on number of worker processes",
    )
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
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate immediately after build",
    )
    build_parser.add_argument(
        "--consolidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Consolidate TileDB objects after build",
    )
    build_parser.add_argument(
        "--dataset_id_blocklist_uri",
        help="Dataset blocklist URI",
        default=default_config.dataset_id_blocklist_uri,
    )
    # hidden option for testing by devs. Will process only the first 'n' datasets
    build_parser.add_argument("--test-first-n", type=int, default=0)
    # hidden option for testing by devs. Allow for WIP testing by devs.
    build_parser.add_argument("--disable-dirty-git-check", action=argparse.BooleanOptionalAction, default=False)

    # VALIDATE
    subparsers.add_parser("validate", help="Validate an existing Census build")

    return parser


if __name__ == "__main__":
    sys.exit(main())
