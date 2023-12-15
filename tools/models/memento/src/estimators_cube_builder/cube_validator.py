import os
import sys

import tiledb

from tools.models.memento.src.estimators_cube_builder.cube_schema import (
    CUBE_LOGICAL_DIMS,
    CUBE_LOGICAL_DIMS_OBS,
)


def _validate_dim_group_uniqueness(cube: tiledb.SparseArray) -> None:
    # retrieve all logical dimension columns, and ignore the estimator columns
    dimensions_df = cube.df[:].reset_index()[CUBE_LOGICAL_DIMS]
    group_counts = dimensions_df.value_counts()
    assert all(group_counts <= 1), "duplicate dimension groups found"


def _validate_all_obs_dims_groups_present(cube: tiledb.SparseArray, source_obs: tiledb.SparseArray) -> None:
    distinct_obs_dims_df = source_obs.df[:][CUBE_LOGICAL_DIMS_OBS].drop_duplicates().set_index(CUBE_LOGICAL_DIMS_OBS)
    distinct_cube_dims_df = cube.df[:][CUBE_LOGICAL_DIMS_OBS].drop_duplicates().set_index(CUBE_LOGICAL_DIMS_OBS)
    actual = set(distinct_cube_dims_df.index)
    expected = set(distinct_obs_dims_df.index)
    missing = expected.difference(actual)
    # Note: This should not fail if the Experiment includes all of the Census genes. If not all genes are included,
    # then some obs groups may have no X data, and will not be included in the cube.
    assert (
        actual == expected
    ), f"not all obs dimensions groups are present in the cube; missing {len(missing)} groups: {missing}"


def validate_cube(cube_path: str, source_experiment_uri: str) -> bool:
    """
    Validate that the cube at the given path is a valid memento estimators cube.
    """
    with tiledb.open(cube_path, "r") as cube:
        _validate_dim_group_uniqueness(cube)

        with tiledb.open(os.path.join(source_experiment_uri, "obs")) as source_obs:
            _validate_all_obs_dims_groups_present(cube, source_obs)
    return True


if __name__ == "__main__":
    validate_cube(sys.argv[1], sys.argv[2])
