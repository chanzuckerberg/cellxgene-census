from typing import Any

import pytest
import requests_mock as rm
import s3fs

import cell_census
from cell_census._release_directory import CELL_CENSUS_RELEASE_DIRECTORY_URL

DIRECTORY_JSON = {
    "latest": "2022-11-01",
    "2022-11-01": {
        "release_date": "2022-11-30",
        "release_build": "2022-11-01",
        "soma": {
            "uri": "s3://cellxgene-data-public/cell-census/2022-11-01/soma/",
            "s3_region": "us-west-2",
        },
        "h5ads": {
            "uri": "s3://cellxgene-data-public/cell-census/2022-11-01/h5ads/",
            "s3_region": "us-west-2",
        },
    },
    "2022-10-01": {
        "release_date": "2022-10-30",
        "release_build": "2022-10-01",
        "soma": {
            "uri": "s3://cellxgene-data-public/cell-census/2022-10-01/soma/",
            "s3_region": "us-west-2",
        },
        "h5ads": {
            "uri": "s3://cellxgene-data-public/cell-census/2022-10-01/h5ads/",
            "s3_region": "us-west-2",
        },
    },
    # An explicitly dangling tag, to confirm we handle correct
    # Underscore indicates expected failure to test below
    "_dangling": "no-such-tag",
}


@pytest.fixture
def directory_mock(requests_mock: rm.Mocker) -> Any:
    return requests_mock.get(CELL_CENSUS_RELEASE_DIRECTORY_URL, json=DIRECTORY_JSON)


def test_get_census_version_directory(directory_mock: Any) -> None:
    directory = cell_census.get_census_version_directory()

    assert isinstance(directory, dict)
    assert len(directory) > 0
    assert all((type(k) == str for k in directory.keys()))
    assert all((type(v) == dict for v in directory.values()))

    for tag in DIRECTORY_JSON:
        assert tag[0] == "_" or tag in directory
        if isinstance(DIRECTORY_JSON[tag], dict):
            assert directory[tag] == DIRECTORY_JSON[tag]

    assert directory["latest"] == directory["2022-11-01"]

    for tag in directory:
        assert directory[tag] == cell_census.get_census_version_description(tag)


def test_get_census_version_description_errors() -> None:
    with pytest.raises(KeyError):
        cell_census.get_census_version_description(census_version="no/such/version/exists")


@pytest.mark.live_corpus
def test_live_directory_contents() -> None:
    # Sanity check that all directory contents are usable. This uses the
    # live directory, so it _could_ start failing without a code change.
    # But given the purpose of this package, that seems like a reasonable
    # tradeoff, as the data directory should never be "corrupt" or there
    # is widespread impact on users.

    fs = s3fs.S3FileSystem(anon=True, cache_regions=True)

    directory = cell_census.get_census_version_directory()
    assert "latest" in directory

    for version, version_description in directory.items():
        with cell_census.open_soma(census_version=version) as census:
            assert census is not None

        assert fs.exists(version_description["soma"]["uri"])
        assert fs.exists(version_description["h5ads"]["uri"])
