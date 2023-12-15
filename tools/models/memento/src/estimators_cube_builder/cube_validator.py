import os
import sys

import tiledb

from .cube_schema import (
    CUBE_LOGICAL_DIMS,
    CUBE_LOGICAL_DIMS_OBS,
)


def _validate_dim_group_uniqueness(cube: tiledb.SparseArray) -> None:
    # retrieve all logical dimension columns, and ignore the estimator columns
    dimensions_df = cube.df[:].reset_index()[CUBE_LOGICAL_DIMS]
    group_counts = dimensions_df.value_counts()
    assert all(group_counts <= 1), "duplicate dimension groups found"


def _validate_all_obs_dims_groups_present(cube: tiledb.SparseArray, source_obs: tiledb.SparseArray) -> None:
    distinct_obs_dims_df = source_obs.df[:][CUBE_LOGICAL_DIMS_OBS].set_index(CUBE_LOGICAL_DIMS_OBS)
    distinct_cube_dims_df = cube.df[:][CUBE_LOGICAL_DIMS_OBS].drop_duplicates().set_index(CUBE_LOGICAL_DIMS_OBS)
    actual = set(distinct_cube_dims_df.index)
    expected = set(distinct_obs_dims_df.index)
    missing = expected.difference(actual)
    assert (
        actual == expected
    ), f"not all obs dimensions groups are present in the cube; missing {len(missing)} groups: {missing}"


def _validate_n_obs_sum(cube: tiledb.SparseArray, source_obs: tiledb.SparseArray) -> None:
    cube_n_obs_sums = cube.df[:].set_index(CUBE_LOGICAL_DIMS_OBS)[["feature_id", "n_obs"]].groupby(["feature_id"]).sum()
    source_obs_len = source_obs.df[:].shape[0]
    # Sum of n_obs (for each gene) will not generally be equal to the number of rows in the obs table, because
    # not all obs groups will have X data for a given gene and will not be included in the cube. The best we can do is
    # ensure the per-gene n_obs sums are less than the number of rows in the obs table.
    assert all(cube_n_obs_sums < source_obs_len)


def validate_cube(cube_path: str, source_experiment_uri: str) -> bool:
    """
    Validate that the cube at the given path is a valid memento estimators cube.
    """
    with tiledb.open(cube_path, "r") as cube:
        _validate_dim_group_uniqueness(cube)

        with tiledb.open(os.path.join(source_experiment_uri, "obs")) as source_obs:
            _validate_n_obs_sum(cube, source_obs)
            _validate_all_obs_dims_groups_present(cube, source_obs)
    return True


if __name__ == "__main__":
    validate_cube(sys.argv[1], sys.argv[2])
