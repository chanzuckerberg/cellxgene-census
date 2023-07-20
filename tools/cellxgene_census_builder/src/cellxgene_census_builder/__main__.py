import argparse
import logging
import pathlib
import sys
from typing import Callable, List

import s3fs

from . import __version__
from .build_state import CENSUS_BUILD_CONFIG, CENSUS_BUILD_STATE, CensusBuildArgs, CensusBuildConfig, CensusBuildState
from .util import log_process_resource_status, process_init, urlcat


def main() -> int:
    cli_parser = create_args_parser()
    cli_args = cli_parser.parse_args()

    working_dir = pathlib.PosixPath(cli_args.working_dir)
    if not working_dir.is_dir():
        logging.critical("Census builder: unable to find working directory - exiting.")
        return 1

    if (working_dir / CENSUS_BUILD_CONFIG).is_file():
        build_config = CensusBuildConfig.load(working_dir / CENSUS_BUILD_CONFIG)
    else:
        build_config = CensusBuildConfig()

    if not cli_args.test_resume:
        if (working_dir / CENSUS_BUILD_STATE).exists():
            logging.critical("Found pre-existing census build in working directory - aborting census build.")
            return 1
        build_state = CensusBuildState()
    else:
        build_state = CensusBuildState.load(working_dir / CENSUS_BUILD_STATE)

    build_args = CensusBuildArgs(working_dir=working_dir, config=build_config, state=build_state)

    # Process initialization/setup must be done early. NOTE: do NOT log before this line!
    process_init(build_args)

    # Return process exit code (or raise, which exits with a code of `1`)
    return do_build(build_args, skip_completed_steps=cli_args.test_resume)


def do_build(args: CensusBuildArgs, skip_completed_steps: bool = False) -> int:
    """
    Top-level build sequence.

    Built steps will be executed in order. Build will stop if a build step returns non-zero
    exit code or raises.
    """
    logging.info(f"Census build: start [version={__version__}]")
    logging.info(args)

    build_steps: List[Callable[[CensusBuildArgs], bool]] = [
        do_prebuild_set_defaults,
        do_prebuild_checks,
        do_build_soma,
        do_validate_soma,
        do_create_reports,
        do_data_copy,
        do_the_release,
        do_report_copy,
        do_old_release_cleanup,
    ]
    try:
        for n, build_step in enumerate(build_steps, start=1):
            step_n_of = f"Build step {build_step.__name__} [{n} of {len(build_steps)}]"
            if skip_completed_steps and args.state.get(build_step.__name__):
                logging.info(f"{step_n_of}: already complete, skipping.")
                continue

            logging.info(f"{step_n_of}: start")
            if not build_step(args):
                logging.critical(f"{step_n_of}: failed, aborting build.")
                return 1

            args.state[build_step.__name__] = True
            args.state.commit(args.working_dir / CENSUS_BUILD_STATE)
            logging.info(f"{step_n_of}: complete")

        logging.info("Census build: completed")

        # And last, last, last ... stash the logs
        do_log_copy(args)

    except Exception:
        logging.critical("Caught exception, exiting", exc_info=True)
        return 1

    log_process_resource_status(level=logging.INFO)
    return 0


def do_prebuild_set_defaults(args: CensusBuildArgs) -> bool:
    """Set any defaults required by build steps."""
    return True


def do_prebuild_checks(args: CensusBuildArgs) -> bool:
    """Pre-build checks for host, config, etc. All pre-conditions should go here."""
    from .host_validation import check_host

    # check host configuration, e.g., free disk space
    if not check_host(args):
        return False

    # verify the build tag is not already published/in use
    build_tag = args.config.build_tag
    assert build_tag is not None
    s3path = urlcat(args.config.cellxgene_census_S3_path, build_tag)
    if s3fs.S3FileSystem(anon=False).exists(s3path):
        logging.error(f"Build tag {build_tag} already exists at {s3path}.")
        return False

    return True


def do_build_soma(args: CensusBuildArgs) -> bool:
    from .build_soma import build as build_a_soma

    if (cc := build_a_soma(args)) != 0:
        logging.critical(f"Build of census failed with code {cc}.")
        return False

    return True


def do_validate_soma(args: CensusBuildArgs) -> bool:
    from .build_soma import validate as validate_a_soma

    if not validate_a_soma(args):
        logging.critical("Validation of the census build has failed.")
        return False

    return True


def do_create_reports(args: CensusBuildArgs) -> bool:
    from .census_summary import display_diff, display_summary

    reports_dir = args.working_dir / args.config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Creating summary report")
    with open(reports_dir / f"census-summary-{args.build_tag}.txt", mode="w") as f:
        display_summary(uri=args.soma_path.as_posix(), file=f)

    logging.info("Creating diff report (new build vs 'latest')")
    with open(reports_dir / f"census-diff-{args.build_tag}.txt", mode="w") as f:
        display_diff(uri=args.soma_path.as_posix(), previous_census_version="latest", file=f)

    return True


def do_data_copy(args: CensusBuildArgs) -> bool:
    """Copy data to S3, in preparation for a release"""
    from .data_copy import sync_to_S3

    sync_to_S3(
        args.working_dir / args.build_tag,
        urlcat(args.config.cellxgene_census_S3_path, args.build_tag),
        dryrun=args.config.dryrun,
    )
    return True


def do_the_release(args: CensusBuildArgs) -> bool:
    from .release_manifest import CensusVersionDescription, make_a_release

    release: CensusVersionDescription = {
        "release_date": None,
        "release_build": args.build_tag,
        "soma": {
            "uri": urlcat(args.config.cellxgene_census_S3_path, args.build_tag, "soma/"),
            "s3_region": "us-west-2",
        },
        "h5ads": {
            "uri": urlcat(args.config.cellxgene_census_S3_path, args.build_tag, "h5ads/"),
            "s3_region": "us-west-2",
        },
        "do_not_delete": False,
    }
    make_a_release(
        args.config.cellxgene_census_S3_path, args.build_tag, release, make_latest=True, dryrun=args.config.dryrun
    )
    return True


def do_report_copy(args: CensusBuildArgs) -> bool:
    """Copy build summary reports to S3 for posterity."""
    from .data_copy import sync_to_S3

    sync_to_S3(
        args.working_dir / args.config.reports_dir,
        urlcat(args.config.logs_S3_path, args.build_tag, args.config.reports_dir),
        dryrun=args.config.dryrun,
    )
    return True


def do_old_release_cleanup(args: CensusBuildArgs) -> bool:
    """Clean up old releases"""
    from .release_cleanup import remove_releases_older_than

    remove_releases_older_than(
        days=args.config.release_cleanup_days,
        census_base_url=args.config.cellxgene_census_S3_path,
        dryrun=args.config.dryrun,
    )
    return True


def do_log_copy(args: CensusBuildArgs) -> bool:
    """Copy logs to S3 for posterity.  Should be the final step, to capture full output of build"""
    from .data_copy import sync_to_S3

    sync_to_S3(
        args.working_dir / args.config.log_dir,
        urlcat(args.config.logs_S3_path, args.build_tag, args.config.log_dir),
        dryrun=args.config.dryrun,
    )
    return True


def create_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cellxgene_census_builder", description="Build the official Census.")
    parser.add_argument("working_dir", type=str, help="Working directory for the build")
    parser.add_argument(
        "--test-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attempt to resume the build by skipping completed workflow steps. CAUTION: TEST OPTION ONLY.",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
