from typing import Any

import pytest
import requests_mock as rm
import s3fs

import cellxgene_census
from cellxgene_census._release_directory import (
    CELL_CENSUS_MIRRORS_DIRECTORY_URL,
    CELL_CENSUS_RELEASE_DIRECTORY_URL,
)

# This test fixture contains 3 releases: 1 "latest" and 2 "LTS". Of the "LTS" releases, one is aliased to "stable"
# and one is "retracted", and both are aliased with "V#" aliases. The ordering of the releases is
# explicitly set to verify that the directory is sorted correctly (i.e. they are not in the desired order here).
# There is also a "dangling" tag, to verify that we handle this case correctly.
#
# TODO: Break this into multiple fixtures to test different scenarios
DIRECTORY_JSON = {
    "2022-10-01": {
        "release_date": "2022-10-30",
        "release_build": "2022-10-01",
        "flags": {"lts": True},
        "soma": {
            "uri": "s3://cellxgene-data-public/cell-census/2022-10-01/soma/",
            "s3_region": "us-west-2",
        },
        "h5ads": {
            "uri": "s3://cellxgene-data-public/cell-census/2022-10-01/h5ads/",
            "s3_region": "us-west-2",
        },
    },
    "2022-09-01": {
        "release_date": "2022-09-30",
        "release_build": "2022-09-01",
        "flags": {"lts": True, "retracted": True},
        "do_not_delete": True,
        "retraction": {
            "date": "2022-11-15",
            "reason": "mistakes happen",
            "info_permalink": "http://cellxgene.com/census/apologies",
        },
        "soma": {
            "uri": "s3://cellxgene-data-public/cell-census/2022-09-01/soma/",
            "s3_region": "us-west-2",
        },
        "h5ads": {
            "uri": "s3://cellxgene-data-public/cell-census/2022-09-01/h5ads/",
            "s3_region": "us-west-2",
        },
    },
    # Ordered the latest release to be last, to verify it is explicitly sorted
    "2022-11-01": {
        "release_date": "2022-11-30",
        "release_build": "2022-11-01",
        "do_not_delete": True,
        "soma": {
            "uri": "s3://cellxgene-data-public/cell-census/2022-11-01/soma/",
            "s3_region": "us-west-2",
        },
        "h5ads": {
            "uri": "s3://cellxgene-data-public/cell-census/2022-11-01/h5ads/",
            "s3_region": "us-west-2",
        },
    },
    # An explicitly dangling tag, to confirm we handle correct
    # Underscore indicates expected failure to test below
    "_dangling": "no-such-tag",
    # Aliases placed at bottom, to verify these are explicitly sorted to the top
    "stable": "V2",
    "latest": "2022-11-01",
    "V2": "2022-10-01",
    "V1": "2022-09-01",
}

MIRRORS_JSON = {
    "default": "AWS-S3-us-west-2",
    "AWS-S3-us-west-2": {
        "provider": "S3",
        "base_uri": "s3://cellxgene-data-public/",
        "region": "us-west-2",
    },
}


@pytest.fixture
def directory_mock(requests_mock: rm.Mocker) -> Any:
    return requests_mock.get(CELL_CENSUS_RELEASE_DIRECTORY_URL, json=DIRECTORY_JSON)


@pytest.fixture
def mirrors_mock(requests_mock: rm.Mocker) -> Any:
    return requests_mock.get(CELL_CENSUS_MIRRORS_DIRECTORY_URL, json=MIRRORS_JSON)


def test_get_census_version_directory(directory_mock: Any) -> None:
    directory = cellxgene_census.get_census_version_directory()

    assert isinstance(directory, dict)
    assert len(directory) > 0
    assert all(isinstance(k, str) for k in directory.keys())
    assert all(isinstance(v, dict) for v in directory.values())

    assert "_dangling" not in directory

    assert directory["2022-11-01"] == DIRECTORY_JSON["2022-11-01"]
    assert directory["2022-10-01"] == DIRECTORY_JSON["2022-10-01"]
    assert "2022-09-01" not in directory  # retracted excluded by default

    assert directory["latest"] == DIRECTORY_JSON["2022-11-01"]
    assert directory["stable"] == DIRECTORY_JSON["2022-10-01"]
    assert directory["V2"] == DIRECTORY_JSON["2022-10-01"]
    assert "V1" not in directory  # retracted excluded by default

    for tag in directory:
        assert directory[tag] == cellxgene_census.get_census_version_description(tag)

    # Verify that the directory is sorted according to this criteria:
    # 1. Aliases first
    # 2. Non aliases after, in reverse order
    dir_list = list(directory)
    assert dir_list == ["stable", "latest", "V2", "2022-11-01", "2022-10-01"]


def test_get_census_version_directory__lts_only(directory_mock: Any) -> None:
    directory = cellxgene_census.get_census_version_directory(lts=True)

    assert directory.keys() == {"stable", "V2", "2022-10-01"}


def test_get_census_version_directory__exclude_lts(directory_mock: Any) -> None:
    directory = cellxgene_census.get_census_version_directory(lts=False)

    assert directory.keys() == {"latest", "2022-11-01"}


def test_get_census_version_directory__include_retracted(directory_mock: Any) -> None:
    directory = cellxgene_census.get_census_version_directory(retracted=None)

    assert "V1" in directory
    assert "2022-09-01" in directory


def test_get_census_version_directory__retraction_info(directory_mock: Any) -> None:
    directory = cellxgene_census.get_census_version_directory(retracted=True)

    assert directory["2022-09-01"]["retraction"] == {
        "date": "2022-11-15",
        "reason": "mistakes happen",
        "info_permalink": "http://cellxgene.com/census/apologies",
    }

    assert directory["V1"]["retraction"] == {
        "date": "2022-11-15",
        "reason": "mistakes happen",
        "info_permalink": "http://cellxgene.com/census/apologies",
    }


def test_get_census_version_description_errors() -> None:
    with pytest.raises(ValueError):
        cellxgene_census.get_census_version_description(census_version="no/such/version/exists")


def test_get_census_mirrors_directory(mirrors_mock: Any) -> None:
    directory = cellxgene_census.get_census_mirror_directory()
    assert "default" not in directory
    assert "AWS-S3-us-west-2" in directory
    assert directory["AWS-S3-us-west-2"] == MIRRORS_JSON["AWS-S3-us-west-2"]


@pytest.mark.live_corpus
def test_live_directory_contents() -> None:
    # Sanity check that all directory contents are usable. This uses the
    # live directory, so it _could_ start failing without a code change.
    # But given the purpose of this package, that seems like a reasonable
    # tradeoff, as the data directory should never be "corrupt" or there
    # is widespread impact on users.

    fs = s3fs.S3FileSystem(anon=True, cache_regions=True)

    directory = cellxgene_census.get_census_version_directory()
    assert "latest" in directory

    for version, version_description in directory.items():
        with cellxgene_census.open_soma(census_version=version) as census:
            assert census is not None

        assert fs.exists(version_description["soma"]["uri"])
        assert fs.exists(version_description["h5ads"]["uri"])
