# Copyright (c) 2022-2023 Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.

"""Versioning of Census builds

Methods to retrieve information about versions of the publicly hosted Census object.
"""
import typing
from collections import OrderedDict
from typing import Dict, Literal, Optional, Union, cast

import requests
from typing_extensions import NotRequired, TypedDict

"""
The following types describe the expected directory of Census builds, used
to bootstrap all data location requests.
"""
CensusVersionName = str  # census version name, e.g., "release-99", "2022-10-01-test", etc.
CensusLocator = TypedDict(
    "CensusLocator",
    {
        "uri": str,  # [deprecated: only used in census < 1.6.0] absolute resource URI.
        "relative_uri": str,  # resource URI (relative)
        "s3_region": Optional[str],  # [deprecated: only used in census < 1.6.0] if an S3 URI, has optional region
    },
)
CensusVersionRetraction = TypedDict(
    "CensusVersionRetraction",
    {
        "date": str,  # the date of retraction
        "reason": Optional[str],  # the reason for retraction
        "info_url": Optional[str],  # a permalink to more information
        "replaced_by": Optional[str],  # the census version that replaces this one
    },
)
ReleaseFlag = Literal["lts", "retracted"]
ReleaseFlags = Dict[ReleaseFlag, bool]
CensusVersionDescription = TypedDict(
    "CensusVersionDescription",
    {
        "release_date": Optional[str],  # date of release (deprecated)
        "release_build": str,  # date of build
        "soma": CensusLocator,  # SOMA objects locator
        "h5ads": CensusLocator,  # source H5ADs locator
        "flags": NotRequired[ReleaseFlags],  # flags for the release
        "retraction": NotRequired[CensusVersionRetraction],  # if retracted, details of the retraction
    },
)
CensusDirectory = Dict[CensusVersionName, Union[CensusVersionName, CensusVersionDescription]]

"""
A provider identifies a storage medium for the Census, which can either be a cloud provider or a local file.
A value of "unknown" can be specified if the provider isn't specified - the API will try to determine
the correct configuration based on the URI.
"""
Provider = Literal["S3", "file", "unknown"]

"""
A mirror identifies a location that can host the census artifacts. A dict of available mirrors exists
in the mirrors.json file, and looks like this:

{
    "default": "default-mirror",
    "default-mirror": {
        "provider": "S3",
        "base_uri": "s3://a-public-bucket/",
        "region": "us-west-2"
    }
}

"""
CensusMirrorName = str  # name of the mirror
CensusMirror = TypedDict(
    "CensusMirror",
    {
        "provider": Provider,  # provider of the mirror.
        "base_uri": str,  # base URI for the mirror location, e.g. s3://cellxgene-data-public/
        "region": Optional[str],  # region of the bucket or resource
    },
)

CensusMirrors = Dict[CensusMirrorName, Union[CensusMirrorName, CensusMirror]]

"""
A `ResolvedCensusLocator` represent an absolute location of a Census resource, including the provider info.
It is obtained by resolving a relative location against a specified mirror.
"""
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
        ValueError: if unknown census_version value.

    Lifecycle:
        maturing

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
        raise ValueError(f"Unable to locate Census version: {census_version}.")
    return description


def get_census_version_directory(
    *, lts: Optional[bool] = None, retracted: Optional[bool] = False
) -> Dict[CensusVersionName, CensusVersionDescription]:
    """
    Get the directory of Census versions currently available, optionally filtering by specified flags. If a filtering
    flag is not specified, Census versions will not be filtered by that flag. Defaults to including both "long-term
    stable" (LTS) and weekly Census versions, and excluding retracted versions.

    Params:
        lts: A filtering flag to either include or exclude long-term stable releases in the result. If None, no
         filtering is performed based on this flag. Defaults to None, which includes both LTS and non-LTS (weekly)
         versions.
        retracted: A filtering flag to either include or exclude retracted releases in the result. If None, no
         filtering is performed based on this flag. Defaults to False, which excludes retracted releases in the result.

    Returns:
        A dictionary that contains Census version names and their corresponding descriptions. Census versions are
        always named by their release date (``YYYY-MM-DD``) but may also have aliases. If an alias is specified,
        the Census version will appear multiple times in the dictionary, once under it's release date name,
        and again for each alias. Aliases may be: "stable", "latest", or "V#". The "stable" alias is used for the
        most recent LTS release, the "latest" alias is used for the most recent weekly release, and the "V#" aliases
        are used to identify LTS releases by a sequentially incrementing version number.

    Lifecycle:
        maturing

    See Also:
        :func:`get_census_version_description`: get description by census_version_name.

    Examples:
        Get all LTS and weekly versions, but exclude retracted LTS versions:

        >>> cellxgene_census.get_census_version_directory()
            {
                'stable': {
                    'release_date': None,
                    'release_build': '2022-11-29',
                    'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/soma/',
                             's3_region': 'us-west-2'},
                    'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/h5ads/',
                              's3_region': 'us-west-2'},
                    'flags': {'lts': True, 'retracted': False}
                },
                'latest': {
                    'release_date': None,
                    'release_build': '2022-12-01',
                    'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/soma/',
                             's3_region': 'us-west-2'},
                    'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/h5ads/',
                              's3_region': 'us-west-2'},
                    'flags': {'lts': True, 'retracted': False}
                },
                'V2': {
                    'release_date': None,
                    'release_build': '2022-11-29',
                    'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/soma/',
                             's3_region': 'us-west-2'},
                    'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/h5ads/',
                              's3_region': 'us-west-2'},
                    'flags': {'lts': True, 'retracted': False}
                },
                '2022-12-01': {
                    'release_date': None,
                    'release_build': '2022-12-01',
                    'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/soma/',
                             's3_region': 'us-west-2'},
                    'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/h5ads/',
                              's3_region': 'us-west-2'},
                    'flags': {'lts': False, 'retracted': False}
                },
                '2022-11-29': {
                    'release_date': None,
                    'release_build': '2022-11-29',
                    'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/soma/',
                             's3_region': 'us-west-2'},
                    'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/h5ads/',
                              's3_region': 'us-west-2'},
                    'flags': {'lts': True, 'retracted': False}
                }
            }

        Get only LTS versions that are not retracted:

        >>> cellxgene_census.get_census_version_directory(lts=True)
            {
                'stable': {
                    'release_date': None,
                    'release_build': '2022-11-29',
                    'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/soma/',
                             's3_region': 'us-west-2'},
                    'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/h5ads/',
                              's3_region': 'us-west-2'},
                    'flags': {'lts': True, 'retracted': False}
                },
                'V2': {
                    'release_date': None,
                    'release_build': '2022-11-29',
                    'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/soma/',
                             's3_region': 'us-west-2'},
                    'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/h5ads/',
                              's3_region': 'us-west-2'},
                    'flags': {'lts': True, 'retracted': False}
                },
                '2022-11-29': {
                    'release_date': None,
                    'release_build': '2022-11-29',
                    'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/soma/',
                             's3_region': 'us-west-2'},
                    'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-11-29/h5ads/',
                              's3_region': 'us-west-2'},
                    'flags': {'lts': True, 'retracted': False}
                }
            }

        Get only retracted releases:

        >>> cellxgene_census.get_census_version_directory(retracted=True)
            {
                'V1': {
                    'release_date': None,
                    'release_build': '2022-10-15',
                    'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-10-15/soma/',
                             's3_region': 'us-west-2'},
                    'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-10-15/h5ads/',
                              's3_region': 'us-west-2'},
                    'flags': {'lts': True, 'retracted': True},
                    'retraction': {
                        'date': '2022-10-30',
                        'reason': 'mistakes happen',
                        'info_url': 'http://cellxgene.com/census/errata/v1',
                        'replaced_by': 'V2'
                    },
                },
                '2022-10-15': {
                    'release_date': None,
                    'release_build': '2022-10-15',
                    'soma': {'uri': 's3://cellxgene-data-public/cell-census/2022-10-15/soma/',
                             's3_region': 'us-west-2'},
                    'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2022-10-15/h5ads/',
                              's3_region': 'us-west-2'},
                    'flags': {'lts': True, 'retracted': True},
                    'retraction': {
                        'date': '2022-10-30',
                        'reason': 'mistakes happen',
                        'info_url': 'http://cellxgene.com/census/errata/v1',
                        'replaced_by': 'V2'
                    }
                }
            }
    """
    response = requests.get(CELL_CENSUS_RELEASE_DIRECTORY_URL)
    response.raise_for_status()

    directory: CensusDirectory = cast(CensusDirectory, response.json())
    directory_out: CensusDirectory = {}
    aliases: typing.Set[CensusVersionName] = set()

    # Resolve all aliases for easier use
    for census_version_name in list(directory.keys()):
        # Strings are aliases for other census_version_name
        directory_value = directory[census_version_name]
        alias = None
        while isinstance(directory_value, str):
            alias = directory_value
            # resolve aliases
            if alias not in directory:
                # oops, dangling pointer -- drop original census_version_name
                directory.pop(census_version_name)
                break

            directory_value = directory[alias]

        if alias:
            aliases.add(census_version_name)

        # exclude aliases
        if not isinstance(directory_value, dict):
            continue

        # filter by release flags
        census_version_description = cast(CensusVersionDescription, directory_value)
        release_flags = cast(ReleaseFlags, {"lts": lts, "retracted": retracted})
        admitted = all(
            [
                census_version_description.get("flags", {}).get(flag_name, False) == release_flags[flag_name]
                for flag_name, flag_value in release_flags.items()
                if flag_value is not None
            ]
        )
        if not admitted:
            continue

        directory_out[census_version_name] = census_version_description.copy()

    # Cast is safe, as we have removed all aliases
    unordered_directory = cast(Dict[CensusVersionName, CensusVersionDescription], directory_out)

    # Sort by aliases and release date, descending
    aliased_releases = [(k, v) for k, v in unordered_directory.items() if k in aliases]
    concrete_releases = [(k, v) for k, v in unordered_directory.items() if k not in aliases]
    ordered_directory = OrderedDict()
    # Note: reverse sorting of aliases serendipitously orders the names we happen to use in a desirable manner:
    # "stable", "latest", "V#"). This will require a more explicit ordering if we change alias naming conventions.
    for k, v in sorted(aliased_releases, key=lambda k: k[0], reverse=True) + sorted(
        concrete_releases, key=lambda k: k[0], reverse=True
    ):
        ordered_directory[k] = v

    return ordered_directory


def get_census_mirror_directory() -> Dict[CensusMirrorName, CensusMirror]:
    """
    Get the directory of Census mirrors currently available.

    Returns:
        A dictionary that contains mirror names and their corresponding info,
        like the provider and the region.

    Lifecycle:
        maturing
    """
    mirrors = _get_census_mirrors()
    del mirrors["default"]
    return cast(Dict[CensusMirrorName, CensusMirror], mirrors)


def _get_census_mirrors() -> CensusMirrors:
    response = requests.get(CELL_CENSUS_MIRRORS_DIRECTORY_URL)
    response.raise_for_status()
    return cast(CensusMirrors, response.json())
