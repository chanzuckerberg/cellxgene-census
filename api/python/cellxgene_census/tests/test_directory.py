from typing import Any

import pytest
import requests_mock as rm
import s3fs

import cellxgene_census
from cellxgene_census._release_directory import CELL_CENSUS_RELEASE_DIRECTORY_URL

DIRECTORY_JSON = {
    "stable": "2022-10-01",
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
    directory = cellxgene_census.get_census_version_directory()

    assert isinstance(directory, dict)
    assert len(directory) > 0
    assert all((type(k) == str for k in directory.keys()))
    assert all((type(v) == dict for v in directory.values()))

    assert "_dangling" not in directory

    assert directory["2022-11-01"] == {**DIRECTORY_JSON["2022-11-01"], "alias": None}  # type: ignore
    assert directory["2022-10-01"] == {**DIRECTORY_JSON["2022-10-01"], "alias": None}  # type: ignore

    assert directory["latest"] == {**DIRECTORY_JSON["2022-11-01"], "alias": "latest"}  # type: ignore
    assert directory["stable"] == {**DIRECTORY_JSON["2022-10-01"], "alias": "stable"}  # type: ignore

    for tag in directory:
        assert directory[tag] == cellxgene_census.get_census_version_description(tag)


def test_get_census_version_description_errors() -> None:
    with pytest.raises(KeyError):
        cellxgene_census.get_census_version_description(census_version="no/such/version/exists")


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
