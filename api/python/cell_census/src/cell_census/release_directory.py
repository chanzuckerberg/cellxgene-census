from typing import Dict, Optional, TypedDict, Union, cast

import requests

"""
The following types describe the expected directory of Cell Census builds, used
to bootstrap all data location requests.
"""
CensusReleaseTag = str  # name or version of census, eg, "release-99" or "2022-10-01-test"
CensusLocator = TypedDict(
    "CensusLocator",
    {
        "uri": str,  # resource URI
        "s3_region": Optional[str],  # if an S3 URI, has optional region
    },
)
CensusReleaseDescription = TypedDict(
    "CensusReleaseDescription",
    {
        "release_date": Optional[str],  # date of release, optional
        "release_build": str,  # date of build
        "soma": CensusLocator,
        "h5ads": CensusLocator,
    },
)
CensusDirectory = Dict[CensusReleaseTag, Union[CensusReleaseTag, CensusReleaseDescription]]


# URL for the default top-level directory of all public data, formatted as a CensusDirectory
CELL_CENSUS_RELEASE_DIRECTORY_URL = "https://s3.us-west-2.amazonaws.com/cellxgene-data-public/cell-census/release.json"


def get_release_description(tag: str) -> CensusReleaseDescription:
    """Get release description for given tag. Raises KeyError if unknown tag value."""
    census_directory = get_directory()
    description = census_directory.get(tag, None)
    if description is None:
        raise KeyError(f"Unable to locate cell census version: {tag}.")
    return description


def get_directory() -> Dict[CensusReleaseTag, CensusReleaseDescription]:
    """
    Get the directory of cell census releases available.
    """
    response = requests.get(CELL_CENSUS_RELEASE_DIRECTORY_URL)
    response.raise_for_status()
    directory: CensusDirectory = cast(CensusDirectory, response.json())

    # Resolve all aliases for easier use
    for tag in list(directory.keys()):
        # Strings are aliases for other tags
        points_at = directory[tag]
        while isinstance(points_at, str):
            # resolve aliases
            if points_at not in directory:
                # oops, dangling pointer -- drop original tag
                directory.pop(tag)
                break

            points_at = directory[points_at]

        if isinstance(points_at, dict):
            directory[tag] = points_at

    # Cast is safe, as we have removed all tag aliases
    return cast(Dict[CensusReleaseTag, CensusReleaseDescription], directory)
