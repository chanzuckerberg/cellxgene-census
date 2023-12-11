import os

import tiledbsoma as soma
from somacore import AxisQuery


def subset_census(query: soma.ExperimentAxisQuery, output_base_dir: str) -> None:
    """
    Subset the census cube to the given query, returning a new cube.
    """
    with soma.Experiment.create(uri=output_base_dir) as exp_subset:
        obs_data = query.obs().concat()
        obs = soma.DataFrame.create(os.path.join(output_base_dir, "obs"), schema=obs_data.schema)
        exp_subset.set("obs", obs)
        obs.write(obs_data)

        ms = exp_subset.add_new_collection("ms")
        rna = ms.add_new_collection("RNA", soma.Measurement)

        var_data = query.var().concat()
        var = rna.add_new_dataframe("var", schema=var_data.schema)
        var.write(var_data)

        x_data = query.X(layer_name="raw").tables().concat()
        x_type = x_data.schema.field_by_name("soma_data").type
        rna.add_new_collection("X")
        rna["X"].add_new_sparse_ndarray("raw", type=x_type, shape=(None, None))
        rna.X["raw"].write(x_data)


if __name__ == "__main__":
    context = soma.SOMATileDBContext().replace(
        tiledb_config={
            "soma.init_buffer_bytes": 128 * 1024**2,
            "vfs.s3.region": "us-west-2",
            "vfs.s3.no_sign_request": True,
        }
    )
    with soma.Experiment.open(
        "s3://cellxgene-data-public/cell-census/2023-10-23/soma/census_data/homo_sapiens", context=context
    ) as exp:
        query = exp.axis_query(
            measurement_name="RNA",
            var_query=AxisQuery(value_filter="feature_id in ['ENSG00000000419']"),  # , 'ENSG00000002330']"),
            obs_query=AxisQuery(value_filter="is_primary_data == True and tissue_general == 'embryo'"),
        )
        subset_census(query, "small-census")
