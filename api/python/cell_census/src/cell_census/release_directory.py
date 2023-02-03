from typing import Dict, Optional, Union, cast

import requests
from typing_extensions import TypedDict

"""
The following types describe the expected directory of Cell Census builds, used
to bootstrap all data location requests.
"""
CensusVersionName = str  # census version name, e.g., "release-99", "2022-10-01-test", etc.
CensusLocator = TypedDict(
    "CensusLocator",
    {
        "uri": str,  # resource URI
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
    },
)
CensusDirectory = Dict[CensusVersionName, Union[CensusVersionName, CensusVersionDescription]]


# URL for the default top-level directory of all public data, formatted as a CensusDirectory
CELL_CENSUS_RELEASE_DIRECTORY_URL = "https://s3.us-west-2.amazonaws.com/cellxgene-data-public/cell-census/release.json"


def get_census_version_description(census_version: str) -> CensusVersionDescription:
    """
    Get release description for given census version, from the Cell
    Census release directory. Raises KeyError if unknown census_version
    value [lifecycle: experimental].

    Parameters
    ----------
    census_version : str
        The census version name.

    Returns
    -------
    CensusReleaseDescription
        Dictionary containing a description of the release.

    See Also
    --------
    get_census_version_directory : returns the entire directory as a dict.

    Examples
    --------
    >>> cell_census.get_census_version_description("latest")
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
        raise KeyError(f"Unable to locate cell census version: {census_version}.")
    return description


def get_census_version_directory() -> Dict[CensusVersionName, CensusVersionDescription]:
    """
    Get the directory of cell census releases currently available [lifecycle: experimental].

    Parameters
    ----------
    None

    Returns
    -------
    Dict[CensusReleaseName, CensusReleaseDescription]
        Dictionary of release names and their corresponding
        release description.

    See Also
    --------
    get_census_version_description : get description by census_version.

    Examples
    --------
    >>> cell_census.get_census_version_directory()
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
        while isinstance(points_at, str):
            # resolve aliases
            if points_at not in directory:
                # oops, dangling pointer -- drop original census_version
                directory.pop(census_version)
                break

            points_at = directory[points_at]

        if isinstance(points_at, dict):
            directory[census_version] = points_at

    # Cast is safe, as we have removed all aliases
    return cast(Dict[CensusVersionName, CensusVersionDescription], directory)
