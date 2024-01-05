import datetime
from typing import Type, cast
from unittest import mock

import pytest
from cellxgene_census_builder.build_state import CensusBuildConfig
from cellxgene_census_builder.release_manifest import (
    CENSUS_AWS_REGION,
    CensusLocator,
    CensusReleaseManifest,
    CensusVersionDescription,
    CensusVersionName,
    get_release_manifest,
    make_a_release,
    validate_release_manifest,
)

# Should not be a real URL
TEST_CENSUS_BASE_URL = "s3://bucket/path/"
TEST_CENSUS_BASE_PREFIX = "/path/"


@pytest.mark.live_corpus
def test_get_release_manifest() -> None:
    # The release manifest is read from the primary bucket.
    census_primary_bucket_base_location = CensusBuildConfig().cellxgene_census_S3_path
    # The base (absolute) URI should be the default mirror.
    census_base_url = CensusBuildConfig().cellxgene_census_default_mirror_S3_path
    release_manifest = get_release_manifest(census_primary_bucket_base_location, s3_anon=True)
    assert len(release_manifest) > 0
    assert "latest" in release_manifest
    assert release_manifest["latest"] in release_manifest
    validate_release_manifest(census_base_url, release_manifest, s3_anon=True)


def soma_locator(tag: CensusVersionName) -> CensusLocator:
    return {
        "uri": f"{TEST_CENSUS_BASE_URL}{tag}/soma/",
        "relative_uri": f"{TEST_CENSUS_BASE_PREFIX}{tag}/soma/",
        "s3_region": CENSUS_AWS_REGION,
    }


def h5ads_locator(tag: CensusVersionName) -> CensusLocator:
    return {
        "uri": f"{TEST_CENSUS_BASE_URL}{tag}/h5ads/",
        "relative_uri": f"{TEST_CENSUS_BASE_PREFIX}{tag}/h5ads/",
        "s3_region": CENSUS_AWS_REGION,
    }


@pytest.mark.parametrize(
    "release_manifest",
    [
        {
            "latest": "2022-01-10",
            "2022-01-10": {
                "release_build": "2022-01-10",
                "soma": soma_locator("2022-01-10"),
                "h5ads": h5ads_locator("2022-01-10"),
                "do_not_delete": False,
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
                        "do_not_delete": False,
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


@pytest.mark.parametrize(
    "release_manifest,rls_tag,rls_info,make_latest,expected_new_manifest",
    [
        (
            {
                "latest": "2022-01-10",
                "2022-01-10": {
                    "release_build": "2022-01-10",
                    "soma": soma_locator("2022-01-10"),
                    "h5ads": h5ads_locator("2022-01-10"),
                },
            },
            "2022-02-03",
            {
                "release_build": "2022-02-03",
                "soma": soma_locator("2022-02-03"),
                "h5ads": h5ads_locator("2022-02-03"),
            },
            True,
            {
                "latest": "2022-02-03",
                "2022-01-10": {
                    "release_build": "2022-01-10",
                    "soma": soma_locator("2022-01-10"),
                    "h5ads": h5ads_locator("2022-01-10"),
                },
                "2022-02-03": {
                    "release_build": "2022-02-03",
                    "soma": soma_locator("2022-02-03"),
                    "h5ads": h5ads_locator("2022-02-03"),
                },
            },
        ),
        (
            {
                "latest": "2022-01-10",
                "2022-01-10": {
                    "release_build": "2022-01-10",
                    "soma": soma_locator("2022-01-10"),
                    "h5ads": h5ads_locator("2022-01-10"),
                },
            },
            "2022-02-03",
            {
                "release_build": "2022-02-03",
                "soma": soma_locator("2022-02-03"),
                "h5ads": h5ads_locator("2022-02-03"),
            },
            False,
            {
                "latest": "2022-01-10",
                "2022-01-10": {
                    "release_build": "2022-01-10",
                    "soma": soma_locator("2022-01-10"),
                    "h5ads": h5ads_locator("2022-01-10"),
                },
                "2022-02-03": {
                    "release_build": "2022-02-03",
                    "soma": soma_locator("2022-02-03"),
                    "h5ads": h5ads_locator("2022-02-03"),
                },
            },
        ),
    ],
)
@pytest.mark.parametrize("dryrun", [True, False])
def test_make_a_release(
    release_manifest: CensusReleaseManifest,
    rls_tag: CensusVersionName,
    rls_info: CensusVersionDescription,
    expected_new_manifest: CensusReleaseManifest,
    make_latest: bool,
    dryrun: bool,
) -> None:
    with (
        mock.patch(
            "cellxgene_census_builder.release_manifest.get_release_manifest", return_value=release_manifest.copy()
        ) as get_release_manifest_patch,
        mock.patch("s3fs.S3FileSystem.isdir", return_value=True),
        mock.patch(
            "cellxgene_census_builder.release_manifest._overwrite_release_manifest"
        ) as commit_release_manifest_patch,
    ):
        make_a_release(TEST_CENSUS_BASE_URL, rls_tag, rls_info, make_latest, dryrun=dryrun)
        assert get_release_manifest_patch.called
        if dryrun:
            assert commit_release_manifest_patch.call_count == 0
        else:
            assert commit_release_manifest_patch.call_count == 1
            assert commit_release_manifest_patch.call_args == ((TEST_CENSUS_BASE_URL, expected_new_manifest),)
