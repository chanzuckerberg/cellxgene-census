import pathlib
from typing import Callable
from unittest import mock

import pytest
from cellxgene_census_builder.__main__ import do_data_copy, do_log_copy, do_report_copy, do_the_release
from cellxgene_census_builder.build_state import CensusBuildArgs, CensusBuildConfig
from cellxgene_census_builder.release_manifest import CensusVersionDescription

TEST_BUCKET_PATH = "s3://bucket/path"
TEST_BUCKET_PREFIX = "/path"


@pytest.mark.parametrize("dryrun", [True, False])
@pytest.mark.parametrize("build_tag", ["2020-10-20", "1999-12-31"])
def test_do_data_copy(tmp_path: pathlib.Path, build_tag: str, dryrun: bool) -> None:
    args = CensusBuildArgs(
        working_dir=tmp_path,
        config=CensusBuildConfig(
            build_tag=build_tag,
            dryrun=dryrun,
            cellxgene_census_S3_path=TEST_BUCKET_PATH,
        ),
    )
    from_path = tmp_path / build_tag
    from_path.mkdir(exist_ok=True, parents=True)
    to_path = f"{TEST_BUCKET_PATH}/{build_tag}"

    with mock.patch("subprocess.Popen") as popen_patch:
        popen_patch.return_value.__enter__.return_value.returncode = 0
        popen_patch.return_value.__enter__.return_value.stdout = None
        do_data_copy(args)

    assert popen_patch.call_count == 1
    expect = ["aws", "s3", "sync", from_path.as_posix(), to_path, "--no-progress"]
    if dryrun:
        expect += ["--dryrun"]
    assert popen_patch.call_args[0][0] == expect


@pytest.mark.parametrize("step_func,dir_name", [(do_report_copy, "reports"), (do_log_copy, "logs")])
@pytest.mark.parametrize("dryrun", [True, False])
@pytest.mark.parametrize("build_tag", ["2020-10-20", "1999-12-31"])
def test_do_other_copy(
    tmp_path: pathlib.Path, build_tag: str, dryrun: bool, step_func: Callable[[CensusBuildArgs], None], dir_name: str
) -> None:
    args = CensusBuildArgs(
        working_dir=tmp_path,
        config=CensusBuildConfig(
            build_tag=build_tag,
            dryrun=dryrun,
            logs_S3_path=TEST_BUCKET_PATH,
        ),
    )
    from_path = tmp_path / dir_name
    from_path.mkdir(exist_ok=True, parents=True)
    to_path = f"{TEST_BUCKET_PATH}/{build_tag}/{dir_name}"

    with mock.patch("subprocess.Popen") as popen_patch:
        popen_patch.return_value.__enter__.return_value.returncode = 0
        popen_patch.return_value.__enter__.return_value.stdout = None
        step_func(args)

    assert popen_patch.call_count == 1
    expect = ["aws", "s3", "sync", from_path.as_posix(), to_path, "--no-progress"]
    if dryrun:
        expect += ["--dryrun"]
    assert popen_patch.call_args[0][0] == expect


@pytest.mark.parametrize("dryrun", [True, False])
def test_do_the_release(tmp_path: pathlib.Path, dryrun: bool) -> None:
    build_tag = "2020-02-03"
    args = CensusBuildArgs(
        working_dir=tmp_path,
        config=CensusBuildConfig(
            build_tag=build_tag,
            cellxgene_census_S3_path=TEST_BUCKET_PATH,
            cellxgene_census_default_mirror_S3_path=TEST_BUCKET_PATH,
            dryrun=dryrun,
        ),
    )

    with mock.patch("cellxgene_census_builder.release_manifest.make_a_release") as make_a_release_patch:
        do_the_release(args)

    expected_rls_description: CensusVersionDescription = {
        "release_build": build_tag,
        "release_date": None,
        "soma": {
            "uri": f"{TEST_BUCKET_PATH}/{build_tag}/soma/",
            "relative_uri": f"{TEST_BUCKET_PREFIX}/{build_tag}/soma/",
            "s3_region": "us-west-2",
        },
        "h5ads": {
            "uri": f"{TEST_BUCKET_PATH}/{build_tag}/h5ads/",
            "relative_uri": f"{TEST_BUCKET_PREFIX}/{build_tag}/h5ads/",
            "s3_region": "us-west-2",
        },
        "do_not_delete": False,
    }
    assert make_a_release_patch.call_count == 1
    assert make_a_release_patch.call_args == (
        (TEST_BUCKET_PATH, build_tag, expected_rls_description),
        {"make_latest": True, "dryrun": dryrun},
    )
