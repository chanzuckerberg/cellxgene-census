# Copyright (c) 2022-2023 Chan Zuckerberg Initiative
#
# Licensed under the MIT License.

"""Versioning of Census builds

Methods to retrieve information about versions of the publicly hosted Census object.
"""
from typing import Dict, Optional, Union, cast

import requests
from typing_extensions import TypedDict

from ._util import _uri_join

SUPPORTED_PROVIDERS = ["S3", "file"]

"""
The following types describe the expected directory of Census builds, used
to bootstrap all data location requests.
"""
CensusVersionName = str  # census version name, e.g., "release-99", "2022-10-01-test", etc.
CensusLocator = TypedDict(
    "CensusLocator",
    {
        "uri": str,  # resource URI
        "relative_uri": str,  # resource URI (relative)
        "s3_region": Optional[str],  # if an S3 URI, has optional region
    },
)
CensusVersionDescription = TypedDict(
    "CensusVersionDescription",
    {
        "release_date": Optional[str],  # date of release, optional
        "release_build": str,  # date of build
        "soma": CensusLocator,  # SOMA objects locator
        "h5ads": CensusLocator,  # source H5ADs locator
        "alias": Optional[str],  # the alias of this entry
    },
)
CensusDirectory = Dict[CensusVersionName, Union[CensusVersionName, CensusVersionDescription]]

CensusMirrorName = str  # name of the mirror
CensusMirror = TypedDict(
    "CensusMirror",
    {
        "provider": str,  # provider of the mirror. Only S3 is supported in this version.
        "base_uri": str,  # name of the bucket or resource
        "region": Optional[str],  # region of the bucket or resource
    },
)

CensusMirrors = Dict[CensusMirrorName, Union[CensusMirrorName, CensusMirror]]

ResolvedCensusLocator = TypedDict(
    "ResolvedCensusLocator",
    {
        "uri": str,  # resource URI (absolute)
        "region": Optional[str],  # if an S3 URI, has optional region
        "provider": str,  # Provider
    },
)


# URL for the default top-level directory of all public data
CELL_CENSUS_RELEASE_DIRECTORY_URL = "https://census.cellxgene.cziscience.com/cellxgene-census/v1/release.json"
CELL_CENSUS_MIRRORS_DIRECTORY_URL = "https://census.cellxgene.cziscience.com/cellxgene-census/v1/mirrors.json"


def get_census_version_description(census_version: str) -> CensusVersionDescription:
    """Get release description for given Census version, from the Census release directory.

    Args:
        census_version:
            The census version name.

    Returns:
        ``CensusVersionDescription`` - a dictionary containing a description of the release.

    Raises:
        KeyError: if unknown census_version value.

    Lifecycle:
        Experimental.

    See Also:
        :func:`get_census_version_directory`: returns the entire directory as a dict.

    Examples:
        >>> cellxgene_census.get_census_version_description("latest")
        {'release_date': None,
        'release_build': '2022-12-01',
        'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/soma/',
        's3_region': 'us-west-2'},
        'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/h5ads/',
        's3_region': 'us-west-2'}}
    """
    census_directory = get_census_version_directory()
    description = census_directory.get(census_version, None)
    if description is None:
        raise KeyError(f"Unable to locate Census version: {census_version}.")
    return description


def get_census_version_directory() -> Dict[CensusVersionName, CensusVersionDescription]:
    """
    Get the directory of Census releases currently available.

    Returns:
        A dictionary that contains release names and their corresponding release description.

    Lifecycle:
        Experimental.

    See Also:
        :func:`get_census_version_description`: get description by census_version.

    Examples:
        >>> cellxgene_census.get_census_version_directory()
        {'latest': {'release_date': None,
        'release_build': '2022-12-01',
        'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/soma/',
        's3_region': 'us-west-2'},
        'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/h5ads/',
        's3_region': 'us-west-2'}},
        '2022-12-01': {'release_date': None,
        'release_build': '2022-12-01',
        'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/soma/',
        's3_region': 'us-west-2'},
        'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/h5ads/',
        's3_region': 'us-west-2'}},
        '2022-11-29': {'release_date': None,
        'release_build': '2022-11-29',
        'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/soma/',
        's3_region': 'us-west-2'},
        'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/h5ads/',
        's3_region': 'us-west-2'}}}
    """
    response = requests.get(CELL_CENSUS_RELEASE_DIRECTORY_URL)
    response.raise_for_status()
    directory: CensusDirectory = cast(CensusDirectory, response.json())

    # Resolve all aliases for easier use
    for census_version in list(directory.keys()):
        # Strings are aliases for other census_version
        points_at = directory[census_version]
        alias = census_version if isinstance(points_at, str) else None
        while isinstance(points_at, str):
            # resolve aliases
            if points_at not in directory:
                # oops, dangling pointer -- drop original census_version
                directory.pop(census_version)
                break

            points_at = directory[points_at]

        if isinstance(points_at, dict):
            directory[census_version] = points_at.copy()
            cast(CensusVersionDescription, directory[census_version])["alias"] = alias

    # Cast is safe, as we have removed all aliases
    return cast(Dict[CensusVersionName, CensusVersionDescription], directory)


def get_census_mirrors() -> CensusMirrors:
    response = requests.get(CELL_CENSUS_MIRRORS_DIRECTORY_URL)
    response.raise_for_status()
    return cast(CensusMirrors, response.json())


def _assert_mirror_supported(mirror: CensusMirror) -> None:
    """
    Verifies if the mirror is supported by this version of the census.
    This method provides a proper error message in case an old version of the census
    tries to connect to an unsupported mirror.
    """
    if mirror["provider"] not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported mirror provider: {mirror['provider']}. Try upgrading the census package.")


def _resolve_census_locator(locator: CensusLocator, mirror: CensusMirror) -> ResolvedCensusLocator:
    _assert_mirror_supported(mirror)

    if locator.get("relative_uri"):
        uri = _uri_join(mirror["base_uri"], locator["relative_uri"])
        region = mirror["region"]
    else:
        uri = locator["uri"]
        region = locator.get("s3_region")
    return ResolvedCensusLocator(uri=uri, region=region, provider=mirror["provider"])
