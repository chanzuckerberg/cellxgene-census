import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import List

import s3fs

from .release_manifest import (
    CensusDirectory,
    CensusVersionName,
    commit_release_manifest,
    get_release_manifest,
    validate_release_manifest,
)
from .util import urlcat


def remove_releases_older_than(days: int, census_base_url: str, dryrun: bool, s3_anon: bool = False) -> None:
    """
    Remove old releases, commiting the change to release.json.

    Current rules - delete releases where:
    * Tag is a date older than `days` in age
    * Tag is aliased by another tag (e.g., 'latest')
    * There will remain at least one build (keeps _at least_ the newest)

    In other words, it will only delete those releases that are not tagged (e.g., 'latest') and which
    are older than 32 days. And it will never delete all releases.

    Age of release is determined by the release version name, i.e., YYYY-MM-DD. The S3 object date or
    other information is not utilized.
    """

    _logit(f"Delete releases older than {days} days old.", dryrun)

    # Load the release manifest
    release_manifest = get_release_manifest(census_base_url=census_base_url, s3_anon=s3_anon)
    # validate that the manifest is safe & sane
    validate_release_manifest(census_base_url, release_manifest, live_corpus_check=True, s3_anon=s3_anon)
    # select build tags that can be deleted
    rls_tags_to_delete = _find_removal_candidates(release_manifest, days)

    _logit(f"Found {len(rls_tags_to_delete)} releases, older than {days} days, and not otherwise tagged.", dryrun)

    # Exit if no work to do.
    if len(rls_tags_to_delete) > 0:
        # IMPORTANT: commit the changes to release.json before doing the delete, in case this
        # fails for some reason (at which point, the deletions should NOT occur).
        _update_release_manifest(release_manifest, rls_tags_to_delete, census_base_url, dryrun)

        # Now delete the builds.
        for rls_tag in rls_tags_to_delete:
            rls_info = release_manifest[rls_tag]
            assert isinstance(rls_info, dict)
            uri = urlcat(rls_info["soma"]["uri"], "..")
            _perform_recursive_delete(rls_tag, uri, dryrun)


def _logit(msg: str, dryrun: bool) -> None:
    logging.info(f"{'(dryrun) ' if dryrun else ''}{msg}")


def _update_release_manifest(
    release_manifest: CensusDirectory,
    rls_tags_to_delete: list[CensusVersionName],
    census_base_url: str,
    dryrun: bool,
) -> None:
    new_manifest: CensusDirectory = {k: v for k, v in release_manifest.items() if k not in rls_tags_to_delete}
    latest_tag = new_manifest["latest"]
    _logit(f"Commiting updated release.json with latest={latest_tag}", dryrun)
    if not dryrun:
        commit_release_manifest(census_base_url, new_manifest)


def _perform_recursive_delete(rls_tag: CensusVersionName, uri: str, dryrun: bool) -> None:
    """Will raise FileNotFound error if the path does not exist (which should never happen)"""
    _logit(f"Delete census release {rls_tag}: {uri}", dryrun)
    if dryrun:
        return
    s3 = s3fs.S3FileSystem(anon=False)
    s3.rm(uri, recursive=True)


def _find_removal_candidates(release_manifest: CensusDirectory, days_older_than: int) -> List[CensusVersionName]:
    delete_before_date = datetime.now() - timedelta(days=days_older_than)

    # all releases which have a tag aliasing them
    is_aliased = [rls for rls in release_manifest.values() if isinstance(rls, str)]

    candidates = []
    for rls_tag, rls_info in release_manifest.items():
        if isinstance(rls_info, dict) and rls_tag not in is_aliased:
            # candidate for deletion - check timestamp
            rls_build_date = datetime.fromisoformat(rls_info["release_build"])
            if rls_build_date < delete_before_date:
                candidates.append(rls_tag)

    # If _all_ releases are candidates for deletion, keep the most recent
    if len(candidates) == len(release_manifest):
        assert len(is_aliased) == 0  # logic check
        candidates.remove(max(release_manifest.keys()))

    return candidates


def main() -> int:
    description = """Delete Census releases older than a user-specified number of days."""
    epilog = """Example:

    python -m cell_census_builder.release_gc s3://cellxgene-data-public/cell-census/ --days 32 --dryrun
    """
    parser = argparse.ArgumentParser(
        prog="cell_census_summary.release_gc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description,
        epilog=epilog,
    )
    parser.add_argument(
        "census_base_uri", type=str, help="Base URL for the Census, e.g., s3://cellxgene-data-public/cell-census/"
    )
    parser.add_argument("--days", type=int, default=32, help="Delete releases N days older than this. Default: 32")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Display, but do NOT perform actions. Useful for previewing actions. Default: false",
    )
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase logging verbosity")
    args = parser.parse_args()

    # Configure the logger.
    level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose == 1 else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)

    remove_releases_older_than(args.days, args.census_base_uri, args.dryrun)
    return 0


if __name__ == "__main__":
    sys.exit(main())
