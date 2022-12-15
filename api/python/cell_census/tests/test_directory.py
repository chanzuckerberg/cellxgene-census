from typing import Any

import pytest
import requests_mock as rm

import cell_census
from cell_census.release_directory import CELL_CENSUS_RELEASE_DIRECTORY_URL

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
        assert tag in directory
        if isinstance(DIRECTORY_JSON[tag], dict):
            assert directory[tag] == DIRECTORY_JSON[tag]

    assert directory["latest"] == directory["2022-11-01"]

    for tag in directory:
        assert directory[tag] == cell_census.get_census_version_description(tag)
