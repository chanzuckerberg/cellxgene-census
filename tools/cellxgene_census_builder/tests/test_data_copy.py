import pathlib
from unittest import mock

import pytest

from cellxgene_census_builder.data_copy import sync_to_S3, sync_to_S3_remote


@pytest.mark.parametrize("dryrun", (True, False))
def test_sync_to_s3(tmp_path: pathlib.Path, dryrun: bool) -> None:
    """copy local path to S3 path"""
    from_path = tmp_path.as_posix()
    to_path = "s3://bucket/bar"
    with mock.patch("subprocess.Popen") as popen_patch:
        popen_patch.return_value.__enter__.return_value.returncode = 0
        popen_patch.return_value.__enter__.return_value.stdout = None
        sync_to_S3(from_path, to_path, dryrun=dryrun)

    assert popen_patch.call_count == 1
    expect = ["aws", "s3", "sync", from_path, to_path, "--no-progress"]
    if dryrun:
        expect += ["--dryrun"]
    assert popen_patch.call_args[0][0] == expect


@pytest.mark.parametrize("dryrun", (True, False))
@pytest.mark.parametrize("delete", (True, False))
def test_sync_to_s3_remote(tmp_path: pathlib.Path, dryrun: bool, delete: bool) -> None:
    """copy S3 path to S3 path"""

    from_path = "s3://bucketA/foo"
    to_path = "s3://bucketB/bar"
    with mock.patch("subprocess.Popen") as popen_patch:
        popen_patch.return_value.__enter__.return_value.returncode = 0
        popen_patch.return_value.__enter__.return_value.stdout = None
        sync_to_S3_remote(from_path, to_path, delete=delete, dryrun=dryrun)

    assert popen_patch.call_count == 1
    expect = ["aws", "s3", "sync", from_path, to_path, "--no-progress"]
    if dryrun:
        expect += ["--dryrun"]
    if delete:
        expect += ["--delete"]
    assert popen_patch.call_args[0][0] == expect


def test_sync_error_checks(tmp_path: pathlib.Path) -> None:
    # mock Popen to be safe
    with mock.patch("subprocess.Popen") as popen_patch:
        popen_patch.return_value.__enter__.return_value.returncode = 0
        popen_patch.return_value.__enter__.return_value.stdout = None

        with pytest.raises(ValueError, match=r"Local path is not a directory"):
            sync_to_S3("/not/a/dir", "s3://foo/bar/", dryrun=True)
        with pytest.raises(ValueError, match=r"S3_path argument does not appear to be an S3 path"):
            sync_to_S3(tmp_path.as_posix(), "/tmp/bar/", dryrun=True)

        with pytest.raises(ValueError, match=r"S3_path argument does not appear to be an S3 path"):
            sync_to_S3_remote("/not/s3/path", "s3://foo/bar/", dryrun=True)
        with pytest.raises(ValueError, match=r"S3_path argument does not appear to be an S3 path"):
            sync_to_S3_remote("s3://foo/bar/", "/not/s3/path/", dryrun=True)
        with pytest.raises(ValueError, match=r"S3_path argument does not appear to be an S3 path"):
            sync_to_S3_remote("/not/s3/path", "/foo/bar/", dryrun=True)
