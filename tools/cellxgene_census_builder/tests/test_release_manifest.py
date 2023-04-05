import datetime
from typing import Type, cast

import pytest
from cellxgene_census_builder.build_state import CENSUS_CONFIG_DEFAULTS
from cellxgene_census_builder.release_manifest import (
    CENSUS_AWS_REGION,
    CensusLocator,
    CensusReleaseManifest,
    CensusVersionDescription,
    CensusVersionName,
    get_release_manifest,
    validate_release_manifest,
)

from .conftest import has_aws_credentials

# Should not be a real URL
TEST_CENSUS_BASE_URL = "s3://bucket/path/"


@pytest.mark.live_corpus
def test_get_release_manifest() -> None:
    census_base_url = CENSUS_CONFIG_DEFAULTS["cellxgene_census_S3_path"]
    release_manifest = get_release_manifest(census_base_url, s3_anon=True)
    assert len(release_manifest) > 0
    assert "latest" in release_manifest
    assert release_manifest["latest"] in release_manifest
    validate_release_manifest(census_base_url, release_manifest, s3_anon=True)


@pytest.mark.skipif(not has_aws_credentials(), reason="Unable to run without AWS credentials")
def test_get_release_manifest_path() -> None:
    with pytest.raises(OSError):
        get_release_manifest("s3://no-such-bucket/or/path")

    with pytest.raises(FileNotFoundError):
        get_release_manifest("s3://cellxgene-data-public/no/such/base/path/")


def soma_locator(tag: CensusVersionName) -> CensusLocator:
    return {"uri": f"{TEST_CENSUS_BASE_URL}{tag}/soma/", "s3_region": CENSUS_AWS_REGION}


def h5ads_locator(tag: CensusVersionName) -> CensusLocator:
    return {"uri": f"{TEST_CENSUS_BASE_URL}{tag}/h5ads/", "s3_region": CENSUS_AWS_REGION}


@pytest.mark.parametrize(
    "release_manifest",
    [
        {
            "latest": "2022-01-10",
            "2022-01-10": {
                "release_build": "2022-01-10",
                "soma": soma_locator("2022-01-10"),
                "h5ads": h5ads_locator("2022-01-10"),
            },
        },
        {
            "latest": "2022-01-10",
            **{
                tag: cast(
                    CensusVersionDescription,
                    {
                        "release_build": tag,
                        "soma": soma_locator(tag),
                        "h5ads": h5ads_locator(tag),
                    },
                )
                for tag in ["2022-01-10", "2023-09-12"]
            },
        },
    ],
)
def test_validate_release_manifest(release_manifest: CensusReleaseManifest) -> None:
    validate_release_manifest(TEST_CENSUS_BASE_URL, release_manifest, live_corpus_check=False)


@pytest.mark.parametrize(
    "release_manifest,expected_error",
    [
        ({}, ValueError),
        ([], TypeError),
        ({datetime.date(2022, 10, 1): "hi"}, TypeError),
        ({"hi": "dangling-ref"}, ValueError),
        ({"tag": []}, TypeError),
        ({"tag": {"release_build": "tag", "soma": soma_locator("tag")}}, ValueError),
        ({"tag": {"release_build": "tag", "h5ads": h5ads_locator("tag")}}, ValueError),
        ({"tag": {"soma": soma_locator("tag"), "h5ads": h5ads_locator("tag")}}, ValueError),
        (
            {"tag": {"release_build": "different_tag", "soma": soma_locator("tag"), "h5ads": h5ads_locator("tag")}},
            ValueError,
        ),
        (
            {
                "tag": {
                    "release_build": "different_tag",
                    "soma": {"uri": "s3://bad/url/"},
                    "h5ads": h5ads_locator("tag"),
                }
            },
            ValueError,
        ),
        (
            {
                "tag": {
                    "release_build": "different_tag",
                    "soma": soma_locator("tag"),
                    "h5ads": {"uri": "s3://bad/url/"},
                }
            },
            ValueError,
        ),
        (
            {
                "not_latest": "2022-01-10",
                "2022-01-10": {
                    "release_build": "2022-01-10",
                    "soma": soma_locator("2022-01-10"),
                    "h5ads": h5ads_locator("2022-01-10"),
                },
            },
            ValueError,
        ),
    ],
)
def test_validate_release_manifest_errors(
    release_manifest: CensusReleaseManifest, expected_error: Type[BaseException]
) -> None:
    with pytest.raises(expected_error):
        validate_release_manifest(TEST_CENSUS_BASE_URL, release_manifest, live_corpus_check=False)
