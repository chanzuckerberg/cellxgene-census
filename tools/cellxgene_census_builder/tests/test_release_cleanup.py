from datetime import datetime, timedelta
from typing import Any, Dict, Type
from unittest import mock

import pytest
from cellxgene_census_builder.release_cleanup import remove_releases_older_than
from cellxgene_census_builder.release_manifest import CensusReleaseManifest, CensusVersionName


def tag_days_old(days_old: int) -> str:
    return (datetime.now() - timedelta(days=days_old)).astimezone().date().isoformat()


TAG_NOW = tag_days_old(0)
TAG_10D_OLD = tag_days_old(10)
TAG_100D_OLD = tag_days_old(100)

S3_PREFIX = "s3://bucket/path/"

RELEASE_MANIFEST: CensusReleaseManifest = {
    "latest": TAG_NOW,
    TAG_NOW: {
        "release_build": TAG_NOW,
        "soma": {"uri": f"{S3_PREFIX}{TAG_NOW}/soma/", "s3_region": "us-west-2"},
        "h5ads": {"uri": f"{S3_PREFIX}{TAG_NOW}/h5ads/", "s3_region": "us-west-2"},
    },
    TAG_10D_OLD: {
        "release_build": TAG_10D_OLD,
        "soma": {"uri": f"{S3_PREFIX}{TAG_10D_OLD}/soma/", "s3_region": "us-west-2"},
        "h5ads": {"uri": f"{S3_PREFIX}{TAG_10D_OLD}/h5ads/", "s3_region": "us-west-2"},
    },
    TAG_100D_OLD: {
        "release_build": TAG_100D_OLD,
        "soma": {"uri": f"{S3_PREFIX}{TAG_100D_OLD}/soma/", "s3_region": "us-west-2"},
        "h5ads": {"uri": f"{S3_PREFIX}{TAG_100D_OLD}/h5ads/", "s3_region": "us-west-2"},
    },
}


@pytest.mark.parametrize("dryrun", [True, False])
@pytest.mark.parametrize(
    "release_manifest,remove_kwargs,expected_delete_tags",
    [
        (RELEASE_MANIFEST, dict(days=0, census_base_url=S3_PREFIX), (TAG_10D_OLD, TAG_100D_OLD)),
        (RELEASE_MANIFEST, dict(days=9, census_base_url=S3_PREFIX), (TAG_10D_OLD, TAG_100D_OLD)),
        (RELEASE_MANIFEST, dict(days=99, census_base_url=S3_PREFIX), (TAG_100D_OLD,)),
        (RELEASE_MANIFEST, dict(days=999, census_base_url=S3_PREFIX), ()),
        (RELEASE_MANIFEST, dict(days=0, census_base_url=S3_PREFIX), (TAG_10D_OLD, TAG_100D_OLD)),
        (
            {**RELEASE_MANIFEST, "latest": TAG_10D_OLD},
            dict(days=0, census_base_url=S3_PREFIX),
            (TAG_NOW, TAG_100D_OLD),
        ),
        ({**RELEASE_MANIFEST, "latest": TAG_10D_OLD}, dict(days=9, census_base_url=S3_PREFIX), (TAG_100D_OLD,)),
    ],
)
def test_remove_releases_older_than(
    release_manifest: CensusReleaseManifest,
    remove_kwargs: Dict[str, Any],
    dryrun: bool,
    expected_delete_tags: list[CensusVersionName],
) -> None:
    """Test the expected happy paths."""

    expected_delete_calls = [mock.call(f"{S3_PREFIX}{tag}/", recursive=True) for tag in expected_delete_tags]
    expected_new_manifest = release_manifest.copy()
    for tag in expected_delete_tags:
        del expected_new_manifest[tag]

    with (
        mock.patch(
            "cellxgene_census_builder.release_cleanup.get_release_manifest", return_value=release_manifest
        ) as get_release_manifest_patch,
        mock.patch("s3fs.S3FileSystem.isdir", return_value=True),
        mock.patch(
            "cellxgene_census_builder.release_manifest._overwrite_release_manifest"
        ) as commit_release_manifest_patch,
        mock.patch("s3fs.S3FileSystem.rm", return_value=None) as delete_patch,
    ):
        remove_releases_older_than(**remove_kwargs, dryrun=dryrun)
        assert get_release_manifest_patch.call_count == 1
        if dryrun:
            assert delete_patch.call_count == 0
            assert commit_release_manifest_patch.call_count == 0
        else:
            assert delete_patch.mock_calls == expected_delete_calls
            if len(expected_delete_tags) > 0:
                assert commit_release_manifest_patch.call_count == 1
                assert commit_release_manifest_patch.call_args == ((S3_PREFIX, expected_new_manifest),)


@pytest.mark.parametrize(
    "release_manifest,remove_kwargs,expected_error",
    [
        # base path check
        (RELEASE_MANIFEST, dict(days=0, census_base_url="s3://not/the/right/path/", dryrun=False), ValueError),
        # check that soma/h5ads are in the same 'directory'
        (
            {
                "latest": TAG_NOW,
                TAG_NOW: {
                    "release_build": TAG_NOW,
                    "soma": {"uri": f"{S3_PREFIX}{TAG_NOW}/soma/"},
                    "h5ads": {"uri": f"{S3_PREFIX}{TAG_NOW}/h5ads/oops/"},
                },
            },
            dict(days=0, census_base_url=S3_PREFIX, dryrun=False),
            ValueError,
        ),
    ],
)
def test_remove_releases_older_than_sanity_checks(
    release_manifest: CensusReleaseManifest, remove_kwargs: Dict[str, Any], expected_error: Type[Exception]
) -> None:
    """Test the expected sanity/error checks"""
    with (
        mock.patch("cellxgene_census_builder.release_cleanup.get_release_manifest", return_value=release_manifest),
        mock.patch("s3fs.S3FileSystem.isdir", return_value=True),
        mock.patch("cellxgene_census_builder.release_manifest._overwrite_release_manifest"),
        mock.patch("s3fs.S3File.write", return_value=0),  # just being paranoid!
        mock.patch("s3fs.S3File.flush", return_value=None),  # just being paranoid!
        mock.patch("s3fs.S3FileSystem.rm", return_value=None),  # just being paranoid!
        pytest.raises(expected_error),
    ):
        remove_releases_older_than(**remove_kwargs)
