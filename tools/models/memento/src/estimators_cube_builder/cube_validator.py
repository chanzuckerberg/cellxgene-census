import sys

import tiledb

from tools.models.memento.src.estimators_cube_builder.cube_schema import (
    CUBE_LOGICAL_DIMS,
)


def _validate_dim_group_uniqueness(cube: tiledb.SparseArray) -> None:
    # retrieve all logical dimension columns, and ignore the estimator columns
    dimensions_df = cube.df[:].reset_index()[CUBE_LOGICAL_DIMS]
    group_counts = dimensions_df.value_counts()
    assert all(group_counts <= 1), "duplicate dimension groups found"


def validate_cube(path: str) -> bool:
    """
    Validate that the cube at the given path is a valid memento estimators cube.
    """
    with tiledb.open(path, "r") as cube:
        _validate_dim_group_uniqueness(cube)
    return True


if __name__ == "__main__":
    validate_cube(sys.argv[1])
