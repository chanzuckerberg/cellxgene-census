import os
import sys

import pyarrow as pa
import tiledb
import tiledbsoma as soma
from somacore import AxisQuery


def subset_census(query: soma.ExperimentAxisQuery, output_base_dir: str) -> None:
    """
    Subset the census cube to the given query, returning a new cube.
    """
    with soma.Experiment.create(uri=output_base_dir) as exp_subset:
        x_data = query.X(layer_name="raw").tables().concat()

        obs_data = query.obs().concat()
        # remove obs rows with no X data
        x_soma_dim_0_unique = pa.Table.from_arrays([x_data["soma_dim_0"].unique()], names=["soma_dim_0"])
        obs_data = obs_data.join(x_soma_dim_0_unique, keys="soma_joinid", right_keys="soma_dim_0", join_type="inner")
        obs = soma.DataFrame.create(os.path.join(output_base_dir, "obs"), schema=obs_data.schema)
        obs.write(obs_data)
        exp_subset.set("obs", obs)

        ms = exp_subset.add_new_collection("ms")
        rna = ms.add_new_collection("RNA", soma.Measurement)

        var_data = query.var().concat()
        var = rna.add_new_dataframe("var", schema=var_data.schema)
        var.write(var_data)

        x_type = x_data.schema.field_by_name("soma_data").type
        rna.add_new_collection("X")
        rna["X"].add_new_sparse_ndarray("raw", type=x_type, shape=(None, None))
        rna.X["raw"].write(x_data)


if __name__ == "__main__":
    experiment_uri, obs_value_filter, var_value_filter, output_cube_path = sys.argv[1:5]

    context = soma.SOMATileDBContext().replace(
        tiledb_config={
            "soma.init_buffer_bytes": 128 * 1024**2,
            "vfs.s3.region": "us-west-2",
            "vfs.s3.no_sign_request": "false",
        }
    )
    with soma.Experiment.open(experiment_uri, context=context) as exp:
        query = exp.axis_query(
            measurement_name="RNA",
            obs_query=AxisQuery(value_filter=obs_value_filter),
            var_query=AxisQuery(value_filter=var_value_filter),
        )
        subset_census(query, output_cube_path)

    for array_uri in ["obs", "ms/RNA/var", "ms/RNA/X/raw"]:
        uri = os.path.join(output_cube_path, array_uri)
        tiledb.consolidate(uri)
        tiledb.vacuum(uri)
