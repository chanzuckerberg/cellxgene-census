"""
Tools to manage the release.json manifest file.
"""
import json
from typing import Dict, Optional, Union, cast

import s3fs
from typing_extensions import NotRequired, TypedDict

from .util import urlcat

"""
The release.json schema is a semi-public format, used by all end-user packages.
"""

CensusVersionName = str  # census version name, e.g., "release-99", "2022-10-01-test", etc.
CensusLocator = TypedDict(
    "CensusLocator",
    {
        "uri": str,  # resource URI
        "relative_uri": str,  # relative URI
        "s3_region": Optional[str],  # if an S3 URI, has optional region
    },
)
CensusVersionDescription = TypedDict(
    "CensusVersionDescription",
    {
        "release_date": Optional[str],  # date of release (deprecated)
        "release_build": str,  # date of build
        "soma": CensusLocator,  # SOMA objects locator
        "h5ads": CensusLocator,  # source H5ADs locator
        "do_not_delete": Optional[bool],  # if set, prevents automated deletion
        "flags": NotRequired[Dict[str, bool]],  # flags for the release
        "retraction": NotRequired[Dict[str, str]],  # if retracted, details of the retraction
    },
)
CensusReleaseManifest = Dict[CensusVersionName, Union[CensusVersionName, CensusVersionDescription]]

CENSUS_AWS_REGION = "us-west-2"
CENSUS_RELEASE_FILE = "release.json"

# The following tags MUST be in any "valid" Census release.json.  This list may grow.
REQUIRED_TAGS = [
    "latest",  # default census - used by the Python/R packages
]


def get_release_manifest(census_base_url: str, s3_anon: bool = False) -> CensusReleaseManifest:
    """
    Fetch the census release manifest.

    Args:
        census_base_url:
            The base S3 URL of the Census.

    Returns:
        A `CensusReleaseManifest` containing the current release manifest.
    """
    s3 = s3fs.S3FileSystem(anon=s3_anon)
    with s3.open(urlcat(census_base_url, CENSUS_RELEASE_FILE)) as f:
        return cast(CensusReleaseManifest, json.loads(f.read()))


def commit_release_manifest(
    census_base_url: str, release_manifest: CensusReleaseManifest, dryrun: bool = False
) -> None:
    """
    Write a new release manifest to the Census.
    """
    # Out of an abundance of caution, validate the contents
    validate_release_manifest(census_base_url, release_manifest)
    if not dryrun:
        _overwrite_release_manifest(census_base_url, release_manifest)


def _overwrite_release_manifest(census_base_url: str, release_manifest: CensusReleaseManifest) -> None:
    # This is a stand-alone function for ease of testing/mocking.
    s3 = s3fs.S3FileSystem(anon=False)
    with s3.open(urlcat(census_base_url, CENSUS_RELEASE_FILE), mode="w") as f:
        f.write(json.dumps(release_manifest, indent=2))


def validate_release_manifest(
    census_base_url: str, release_manifest: CensusReleaseManifest, live_corpus_check: bool = True, s3_anon: bool = False
) -> None:
    if not isinstance(release_manifest, dict):
        raise TypeError("Release manifest must be a dictionary")

    if len(release_manifest) == 0:
        raise ValueError("Release manifest is empty")

    for rls_tag, rls_info in release_manifest.items():
        if not isinstance(rls_tag, str):
            raise TypeError("Release tags must be a string")

        if isinstance(rls_info, str):
            # alias
            if rls_info not in release_manifest:
                raise ValueError(f"Release manifest contains undefined tag reference {rls_info}")
        else:
            # record
            _validate_release_info(rls_tag, rls_info, census_base_url)
            if live_corpus_check:
                _validate_exists(rls_info, s3_anon)

    for rls_tag in REQUIRED_TAGS:
        if rls_tag not in release_manifest:
            raise ValueError(f"Release manifest is missing required release tag: {rls_tag}")


def _validate_release_info(
    rls_tag: CensusVersionName, rls_info: CensusVersionDescription, census_base_url: str
) -> None:
    if not isinstance(rls_info, dict):
        raise TypeError("Release records must be a dict")

    if not all(k in rls_info for k in ("release_build", "soma", "h5ads")):
        raise ValueError("Release info is missing required field")

    if rls_info["release_build"] != rls_tag:
        raise ValueError("release_build must be the same as the release tag")

    from urllib.parse import urlparse

    parsed_url = urlparse(census_base_url)
    prefix = parsed_url.path

    expected_soma_locator = {
        "relative_uri": urlcat(prefix, rls_tag, "soma/"),
        "s3_region": CENSUS_AWS_REGION,
    }
    expected_h5ads_locator = {
        "relative_uri": urlcat(prefix, rls_tag, "h5ads/"),
        "s3_region": CENSUS_AWS_REGION,
    }

    # uri (a.k.a. absolute_uri) is legacy and depends on a specific location. To simplify
    # the code, we can skip this check.
    rls_info_soma = dict(rls_info["soma"])
    del rls_info_soma["uri"]
    rls_info_h5ads = dict(rls_info["h5ads"])
    del rls_info_h5ads["uri"]

    if rls_info_soma != expected_soma_locator:
        raise ValueError(f"Release record for {rls_tag} contained unexpected SOMA locator")
    if rls_info_h5ads != expected_h5ads_locator:
        raise ValueError(f"Release record for {rls_tag} contained unexpected H5AD locator")


def _validate_exists(rls_info: CensusVersionDescription, s3_anon: bool) -> None:
    s3 = s3fs.S3FileSystem(anon=s3_anon)

    uri = rls_info["soma"]["uri"]
    if not s3.isdir(uri):
        raise ValueError(f"SOMA URL in release.json does not exist {uri}")
    uri = rls_info["h5ads"]["uri"]
    if not s3.isdir(uri):
        raise ValueError(f"H5ADS URL in release.json does not exist {uri}")


def make_a_release(
    census_base_url: str,
    rls_tag: CensusVersionName,
    rls_info: CensusVersionDescription,
    make_latest: bool,
    dryrun: bool = False,
) -> None:
    """
    Make a release and optionally alias release as `latest`
    """

    manifest = get_release_manifest(census_base_url)
    if rls_tag in manifest:
        raise ValueError(f"Release version {rls_tag} is already in the release manifest")
    manifest[rls_tag] = rls_info

    if make_latest:
        manifest["latest"] = rls_tag

    # Will validate, and raise on anything suspicious
    commit_release_manifest(census_base_url, manifest, dryrun=dryrun)
