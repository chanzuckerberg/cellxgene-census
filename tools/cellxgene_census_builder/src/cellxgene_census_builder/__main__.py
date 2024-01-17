import argparse
import logging
import os
import pathlib
import sys
from typing import Callable, Sequence
from urllib.parse import urlparse

import s3fs

from .build_soma import build as build_a_soma
from .build_soma import validate as validate_a_soma
from .build_state import CENSUS_BUILD_CONFIG, CENSUS_BUILD_STATE, CensusBuildArgs, CensusBuildConfig, CensusBuildState
from .util import log_process_resource_status, process_init, start_resource_logger, urlcat

logger = logging.getLogger(__name__)


def main() -> int:
    cli_parser = create_args_parser()
    cli_args = cli_parser.parse_args()

    working_dir = pathlib.PosixPath(cli_args.working_dir)
    if not working_dir.is_dir():
        logger.critical("Census builder: unable to find working directory - exiting.")
        return 1

    if (working_dir / CENSUS_BUILD_CONFIG).is_file():
        build_config = CensusBuildConfig.load(working_dir / CENSUS_BUILD_CONFIG)
    else:
        build_config = CensusBuildConfig.load_from_env_vars()

    if not cli_args.test_resume:
        if (working_dir / CENSUS_BUILD_STATE).exists():
            logger.critical("Found pre-existing census build in working directory - aborting census build.")
            return 1
        build_state = CensusBuildState()
    else:
        build_state = CensusBuildState.load(working_dir / CENSUS_BUILD_STATE)

    build_args = CensusBuildArgs(working_dir=working_dir, config=build_config, state=build_state)

    # Process initialization/setup must be done early. NOTE: do NOT log before this line!
    process_init(build_args)

    # List of available commands for the builder
    commands = {
        "release": [do_the_release],
        "replicate": [do_sync_to_replica_s3_bucket],
        "sync-release": [
            do_sync_release_file_to_replica_s3_bucket,
        ],
        "build": [
            do_prebuild_set_defaults,
            do_prebuild_checks,
            do_build_soma,
            do_validate_soma,
            do_create_reports,
            do_data_copy,
            do_report_copy,
            do_log_copy,
        ],
        "mock-build": [
            do_mock_build,
            do_data_copy,
        ],
        "cleanup": [
            do_old_release_cleanup,
        ],
        "full-build": [
            do_prebuild_set_defaults,
            do_prebuild_checks,
            do_build_soma,
            do_validate_soma,
            do_create_reports,
            do_data_copy,
            do_the_release,
            do_report_copy,
            do_old_release_cleanup,
            do_log_copy,
        ],
    }

    # The build command is set via the CENSUS_BUILD_COMMAND environment variable.
    command = os.getenv("CENSUS_BUILD_COMMAND")
    if command is None:
        logger.critical("A census command must be specified in the CENSUS_BUILD_COMMAND environment variable.")
        return 1

    # Used for testing.
    if command == "pass":
        return 0

    if command not in commands:
        logger.critical(f"Unknown command: {command}")
        return 1

    start_resource_logger()

    build_steps = commands[command]
    rv = _do_steps(build_steps, build_args, cli_args.test_resume)
    return rv


def _do_steps(
    build_steps: Sequence[Callable[[CensusBuildArgs], bool]], args: CensusBuildArgs, skip_completed_steps: bool = False
) -> int:
    """
    Performs a series of steps as specified by the `build_steps` argument.
    """
    try:
        for n, build_step in enumerate(build_steps, start=1):
            step_n_of = f"Build step {build_step.__name__} [{n} of {len(build_steps)}]"
            if skip_completed_steps and args.state.get(build_step.__name__):
                logger.info(f"{step_n_of}: already complete, skipping.")
                continue

            logger.info(f"{step_n_of}: start")
            if not build_step(args):
                logger.critical(f"{step_n_of}: failed, aborting build.")
                return 1

            args.state[build_step.__name__] = True
            args.state.commit(args.working_dir / CENSUS_BUILD_STATE)
            logger.info(f"{step_n_of}: complete")

    except Exception:
        logger.critical("Caught exception, exiting", exc_info=True)
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
        logger.error(f"Build tag {build_tag} already exists at {s3path}.")
        return False

    return True


def do_build_soma(args: CensusBuildArgs) -> bool:
    if (cc := build_a_soma(args)) != 0:
        logger.critical(f"Build of census failed with code {cc}.")
        return False

    return True


def do_validate_soma(args: CensusBuildArgs) -> bool:
    if not validate_a_soma(args):
        logger.critical("Validation of the census build has failed.")
        return False

    return True


def do_create_reports(args: CensusBuildArgs) -> bool:
    from .census_summary import display_diff, display_summary

    reports_dir = args.working_dir / args.config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating summary report")
    with open(reports_dir / f"census-summary-{args.build_tag}.txt", mode="w") as f:
        display_summary(uri=args.soma_path.as_posix(), file=f)

    logger.info("Creating diff report (new build vs 'latest')")
    with open(reports_dir / f"census-diff-{args.build_tag}.txt", mode="w") as f:
        display_diff(uri=args.soma_path.as_posix(), previous_census_version="latest", file=f)

    return True


def do_mock_build(args: CensusBuildArgs) -> bool:
    """Mock build. Used for testing"""
    args.soma_path.mkdir(parents=True, exist_ok=False)
    args.h5ads_path.mkdir(parents=True, exist_ok=False)
    with open(f"{args.soma_path}/test.soma", "w") as f:
        f.write("test")
    with open(f"{args.h5ads_path}/test.h5ad", "w") as f:
        f.write("test")

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
    """
    Perform the release by publishing changes to the release.json file. Respects `dryrun` flag.
    """

    from .release_manifest import CensusVersionDescription, make_a_release

    parsed_url = urlparse(args.config.cellxgene_census_S3_path)
    prefix = parsed_url.path

    # Absolute URIs are deprecated, but we still need to support them for legacy reasons.
    # They should point to the default mirror location.
    release: CensusVersionDescription = {
        "release_date": None,
        "release_build": args.build_tag,
        "soma": {
            "uri": urlcat(args.config.cellxgene_census_default_mirror_S3_path, args.build_tag, "soma/"),
            "relative_uri": urlcat(prefix, args.build_tag, "soma/"),
            "s3_region": "us-west-2",
        },
        "h5ads": {
            "uri": urlcat(args.config.cellxgene_census_default_mirror_S3_path, args.build_tag, "h5ads/"),
            "relative_uri": urlcat(prefix, args.build_tag, "h5ads/"),
            "s3_region": "us-west-2",
        },
        "do_not_delete": False,
    }
    census_base_url = args.config.cellxgene_census_S3_path
    make_a_release(census_base_url, args.build_tag, release, make_latest=True, dryrun=args.config.dryrun)
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


def do_sync_release_file_to_replica_s3_bucket(args: CensusBuildArgs) -> bool:
    """Copy release.json to replica S3 bucket"""
    from .data_copy import sync_to_S3_remote

    source_key = urlcat(args.config.cellxgene_census_S3_path, args.build_tag, "release.json")
    dest_key = urlcat(args.config.cellxgene_census_S3_replica_path, args.build_tag, "release.json")

    sync_to_S3_remote(
        source_key,
        dest_key,
        dryrun=args.config.dryrun,
    )
    return True


def do_sync_to_replica_s3_bucket(args: CensusBuildArgs) -> bool:
    """
    Sync data to replica S3 bucket. Syncs everything and deletes anything
    in the replica bucket that is not in the primary bucket.
    """
    from .data_copy import sync_to_S3_remote

    sync_to_S3_remote(
        urlcat(args.config.cellxgene_census_S3_path),
        urlcat(args.config.cellxgene_census_S3_replica_path),
        delete=True,
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
