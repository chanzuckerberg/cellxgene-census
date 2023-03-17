import argparse
import multiprocessing
import sys
from datetime import datetime

from .build import build
from .experiment_builder import ExperimentBuilder
from .experiment_specs import make_experiment_specs
from .mp import process_initializer
from .util import uricat
from .validate import validate


def main() -> int:
    parser = create_args_parser()
    args = parser.parse_args()
    assert args.subcommand in ["build", "validate"]

    process_initializer(args.verbose)

    # normalize our base URI - must include trailing slash
    soma_path = uricat(args.uri, args.build_tag, "soma")
    assets_path = uricat(args.uri, args.build_tag, "h5ads")

    # create the experiment specifications and builders
    experiment_specifications = make_experiment_specs()
    experiment_builders = [ExperimentBuilder(spec) for spec in experiment_specifications]

    cc = 0
    if args.subcommand == "build":
        cc = build(args, soma_path, assets_path, experiment_builders)

    if cc == 0 and (args.subcommand == "validate" or args.validate):
        validate(args, soma_path, assets_path, experiment_specifications)

    return cc


def create_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cell_census_builder")
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
    build_parser = subparsers.add_parser("build", help="Build Cell Census")
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
    subparsers.add_parser("validate", help="Validate an existing cell census build")

    return parser


if __name__ == "__main__":
    # this is very important to do early, before any use of `concurrent.futures`
    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)

    sys.exit(main())
