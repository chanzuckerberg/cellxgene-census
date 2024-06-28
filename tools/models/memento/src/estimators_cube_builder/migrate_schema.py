import sys

import tiledb
from cube_schema import ESTIMATOR_NAMES, build_estimators_schema

if __name__ == "__main__":
    old_cube_uri = sys.argv[1]
    new_cube_uri = sys.argv[2]

    tdb_config = tiledb.Config(
        {
            "py.init_buffer_bytes": 1 * 1024**3,
        }
    )

    with tiledb.open(old_cube_uri, "r", config=tdb_config) as old_cube:
        n_obs_groups = old_cube.schema.domain.dim(1).domain[1]
        new_schema = build_estimators_schema(n_obs_groups)
        tiledb.Array.create(new_cube_uri, new_schema, overwrite=False)
        with tiledb.open(new_cube_uri, "w") as new_cube:
            for i, old_chunk in enumerate(
                old_cube.query(return_incomplete=True, use_arrow=True, return_arrow=True, attrs=ESTIMATOR_NAMES).df[:],
                start=1,
            ):
                print(f"writing chunk {i}, shape={old_chunk.shape}")
                coords = [old_chunk[dim.name].combine_chunks() for dim in new_cube.schema.domain]
                data = {attr.name: old_chunk[attr.name].combine_chunks() for attr in new_schema}
                new_cube[tuple(coords)] = data

    print("performing consolidate & vacuum...")
    tiledb.consolidate(new_cube_uri)
    tiledb.vacuum(new_cube_uri)
