from os import path
from tempfile import TemporaryDirectory

import tiledb

from tools.models.memento.src.estimators_cube_builder.cube_builder import build
from tools.models.memento.src.estimators_cube_builder.cube_schema import CUBE_LOGICAL_DIMS


def test_cube_builder_regression() -> None:
    pwd = path.dirname(__file__)
    with TemporaryDirectory() as cube_dir:
        build(
            cube_uri=cube_dir, experiment_uri=path.join(pwd, "fixtures", "census-human-tongue-2genes"), validate=False
        )
        with tiledb.open(cube_dir) as actual_cube, tiledb.open(
            path.join(pwd, "fixtures", "estimators-cube-human-tongue-2genes")
        ) as expected_cube:
            expected_cube_df = expected_cube.df[:].set_index(CUBE_LOGICAL_DIMS).sort_index()
            actual_cube_df = actual_cube.df[:].set_index(CUBE_LOGICAL_DIMS).sort_index()
            assert all(actual_cube_df == expected_cube_df)
