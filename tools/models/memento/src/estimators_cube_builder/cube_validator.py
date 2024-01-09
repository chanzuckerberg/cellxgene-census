import os
import sys

import tiledb

from .cube_schema import (
    OBS_GROUPS_ARRAY,
    OBS_LOGICAL_DIMS,
)


def _validate_dim_group_uniqueness(obs_groups: tiledb.SparseArray) -> None:
    # retrieve all logical dimension columns, and ignore the estimator columns
    dimensions_df = obs_groups.df[:].reset_index()[OBS_LOGICAL_DIMS]
    group_counts = dimensions_df.value_counts()
    assert all(group_counts <= 1), "duplicate dimension groups found"


def _validate_all_obs_dims_groups_present(obs_groups: tiledb.SparseArray, source_obs: tiledb.SparseArray) -> None:
    distinct_obs_dims_df = source_obs.df[:][OBS_LOGICAL_DIMS].set_index(OBS_LOGICAL_DIMS)
    distinct_obs_groups_dims_df = obs_groups.df[:][OBS_LOGICAL_DIMS].set_index(OBS_LOGICAL_DIMS)
    actual = set(distinct_obs_groups_dims_df.index)
    expected = set(distinct_obs_dims_df.index)
    missing = expected.difference(actual)
    assert (
        actual == expected
    ), f"not all obs dimensions groups are present in the cube; missing {len(missing)} groups: {missing}"


def _validate_n_obs_sum(obs_groups: tiledb.SparseArray, source_obs: tiledb.SparseArray) -> None:
    cube_n_obs_sums = obs_groups.df[:][["n_obs"]].sum()
    source_obs_len = source_obs.df[:].shape[0]
    assert all(cube_n_obs_sums == source_obs_len)


def validate_cube(cube_uri: str, source_experiment_uri: str) -> bool:
    """
    Validate that the cube at the given path is a valid memento estimators cube.
    """
    obs_groups_uri = os.path.join(cube_uri, OBS_GROUPS_ARRAY)
    # estimators_uri = os.path.join(cube_uri, ESTIMATORS_ARRAY)

    with tiledb.open(os.path.join(source_experiment_uri, "obs")) as source_obs:
        with tiledb.open(obs_groups_uri, "r") as obs_groups:
            _validate_all_obs_dims_groups_present(obs_groups, source_obs)
            _validate_dim_group_uniqueness(obs_groups)
            _validate_n_obs_sum(obs_groups, source_obs)

    # TODO: Check that all 0 < sem < mean
    # with tiledb.open(estimators_uri, "r") as estimators:
    #     pass

    return True


if __name__ == "__main__":
    validate_cube(sys.argv[1], sys.argv[2])
