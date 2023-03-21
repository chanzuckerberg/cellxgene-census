import argparse
import logging
import pathlib
import sys
from typing import Callable, List

import s3fs

from . import __version__
from .build_state import CENSUS_BUILD_CONFIG, CENSUS_BUILD_STATE, CensusBuildArgs, CensusBuildConfig
from .util import process_init, urlcat


def main() -> int:
    cli_parser = create_args_parser()
    cli_args = cli_parser.parse_args()

    working_dir = pathlib.PosixPath(cli_args.working_dir)
    if not working_dir.is_dir():
        logging.critical("Census builder: unable to find working directory - exiting.")
        return 1
    if (working_dir / CENSUS_BUILD_STATE).exists():
        logging.critical("Found pre-existing census build in working directory - aborting census build.")
        return 1

    if (working_dir / CENSUS_BUILD_CONFIG).is_file():
        build_config = CensusBuildConfig.load(working_dir / CENSUS_BUILD_CONFIG)
    else:
        build_config = CensusBuildConfig()

    build_args = CensusBuildArgs(working_dir=working_dir, config=build_config)

    # Process initialization/setup must be done early
    process_init(build_args)

    # Return process exit code (or raise, which exits with a code of `1`)
    return do_build(build_args)


def do_build(args: CensusBuildArgs) -> int:
    """
    Top-level build sequence.

    Built steps will be executed in order. Build will stop if a build step returns non-zero
    exit code or raises.
    """
    logging.info(f"Census build: start [version={__version__}]")
    build_steps: List[Callable[[CensusBuildArgs], int]] = [
        do_prebuild_set_defaults,
        do_prebuild_checks,
        do_build_soma,
        do_validate_soma,
        do_create_reports,
    ]
    try:
        for n, build_step in enumerate(build_steps, start=1):
            logging.info(f"Build step {build_step.__name__} [{n} of {len(build_steps)}]: start")
            cc = build_step(args)
            args.state.commit(args.working_dir / CENSUS_BUILD_STATE)
            if cc != 0:
                logging.critical(f"Build step {build_step.__name__} returned error code {cc}: aborting build.")
                return cc
            logging.info(f"Build step {build_step.__name__} [{n} of {len(build_steps)}]: complete")

    except Exception:
        logging.critical("Caught exception, exiting", exc_info=True)
        return 1

    logging.info("Census build: completed")
    return 0


def do_prebuild_set_defaults(args: CensusBuildArgs) -> int:
    """Set any default state required by build steps."""
    args.state["do_prebuild_set_defaults"] = True
    return 0


def do_prebuild_checks(args: CensusBuildArgs) -> int:
    """Pre-build checks for host, config, etc. All pre-conditions should go here."""
    from .host_validation import check_host

    # check host configuration, e.g., free disk space
    if not check_host(args):
        return 1

    # verify the build tag is not already published/in use
    build_tag = args.config.build_tag
    assert build_tag is not None
    s3path = urlcat(args.config.cell_census_S3_path, build_tag)
    if s3fs.S3FileSystem(anon=True).exists(s3path):
        logging.error(f"Build tag {build_tag} already exists at {s3path}.")
        return 1

    args.state["do_prebuild_checks"] = True
    return 0


def do_build_soma(args: CensusBuildArgs) -> int:
    from .build_soma import build as build_a_soma

    if (cc := build_a_soma(args)) != 0:
        return cc

    args.state["do_build_soma"] = True
    return 0


def do_validate_soma(args: CensusBuildArgs) -> int:
    from .build_soma import validate as validate_a_soma

    if not validate_a_soma(args):
        logging.critical("Validation of the census build has failed.")
        return 1

    args.state["do_validate_soma"] = True
    return 0


def do_create_reports(args: CensusBuildArgs) -> int:
    from .census_summary import display_diff, display_summary

    reports_dir = args.working_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Creating summary report")
    with open(reports_dir / f"census-summary-{args.build_tag}.txt", mode="w") as f:
        display_summary(uri=args.soma_path.as_posix(), file=f)

    logging.info("Creating diff report (new build vs 'latest')")
    with open(reports_dir / f"census-diff-{args.build_tag}.txt", mode="w") as f:
        display_diff(uri=args.soma_path.as_posix(), previous_census_version="latest", file=f)

    args.state["do_create_reports"] = True
    return 0


def create_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cell_census_builder")
    parser.add_argument("working_dir", type=str, help="Working directory for the build")
    return parser


sys.exit(main())
