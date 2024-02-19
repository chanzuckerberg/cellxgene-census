import argparse
import logging
import pathlib
import subprocess
import sys
from typing import Union

from .logging import logging_init_params

logger = logging.getLogger(__name__)


def sync_to_S3(from_path: Union[str, pathlib.PosixPath], to_path: str, dryrun: bool = False) -> None:
    """Copy (sync) local directory to S3.

    Equivalent of `aws s3 sync local_directory_path S3_path`.

    Raises upon error.
    """
    from_path = pathlib.PosixPath(from_path).absolute()
    if not from_path.is_dir():
        raise ValueError(f"Local path is not a directory: {from_path.as_posix()}")
    if not to_path.startswith("s3://"):
        raise ValueError(f"S3_path argument does not appear to be an S3 path: {to_path}")

    _sync_to_S3(from_path.as_posix(), to_path, delete=False, dryrun=dryrun)


def sync_to_S3_remote(from_path: str, to_path: str, delete: bool = False, dryrun: bool = False) -> None:
    """Copy (sync) between two S3 locations.

    Equivalent of `aws s3 sync S3_path_source S3_path_dst`.

    Raises upon error.
    """
    if not from_path.startswith("s3://") or not to_path.startswith("s3://"):
        raise ValueError(f"S3_path argument does not appear to be an S3 path: {to_path}")

    _sync_to_S3(from_path, to_path, delete=delete, dryrun=dryrun)


def _sync_to_S3(from_path: str, to_path: str, delete: bool = False, dryrun: bool = False) -> None:
    cmd = ["aws", "s3", "sync", from_path, to_path, "--no-progress"]
    if dryrun:
        cmd += ["--dryrun"]
    if delete:
        cmd += ["--delete"]

    returncode = -1
    try:
        _log_it(f"Starting sync {from_path} -> {to_path}, delete: {delete}", dryrun)
        with subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True) as proc:
            print(proc.returncode)
            if proc.stdout is not None:
                for line in proc.stdout:
                    logger.info(line)

        returncode = proc.returncode
        if returncode:
            raise subprocess.CalledProcessError(returncode, proc.args)

    finally:
        _log_it(f"Completed sync, return code {returncode}, {from_path} -> {to_path}, delete: {delete}", dryrun)


def _log_it(msg: str, dryrun: bool) -> None:
    logger.info(f"{'(dryrun) ' if dryrun else ''}{msg}")


def main() -> int:
    description = """Sync (copy) a local directory to an S3 location."""
    epilog = """Example:

    python -m cellxgene_census_builder.data_copy /tmp/data/ s3://bucket/path/ --dryrun
    """
    parser = argparse.ArgumentParser(
        prog="cellxgene_census_builder.data_copy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description,
        epilog=epilog,
    )
    parser.add_argument("from_path", type=str, help="Data source, specified as a local path, e.g., /home/me/files/")
    parser.add_argument("to_path", type=str, help="S3 path data is copied to, e.g., s3://bucket/path/")
    parser.add_argument(
        "--dryrun",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skips S3 data copies. Useful for previewing actions. Default: True",
    )
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase logging verbosity")
    args = parser.parse_args()

    # Configure the logger.
    logging_init_params(args.verbose)

    sync_to_S3(args.from_path, args.to_path, dryrun=args.dryrun)
    return 0


if __name__ == "__main__":
    sys.exit(main())
