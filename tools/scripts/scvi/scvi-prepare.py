import cellxgene_census
import tiledbsoma as soma
import yaml
from cellxgene_census.experimental.pp import highly_variable_genes

file = "scvi-config.yaml"

if __name__ == "__main__":
    with open(file) as f:
        config = yaml.safe_load(f)

    census = cellxgene_census.open_soma(census_version="latest")

    census_config = config.get("census")
    experiment_name = census_config.get("organism")
    obs_query = census_config.get("obs_query")
    obs_query_model = census_config.get("obs_query_model")

    if obs_query is None:
        obs_value_filter = obs_query_model
    else:
        obs_value_filter = f"{obs_query} and {obs_query_model}"

    query = census["census_data"][experiment_name].axis_query(
        measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
    )

    hvg_config = config.get("hvg")
    top_n_hvg = hvg_config.get("top_n_hvg")
    hvg_batch = hvg_config.get("hvg_batch")
    min_genes = hvg_config.get("min_genes")

    print("Starting hvg selection")

    hvgs_df = highly_variable_genes(query, n_top_genes=top_n_hvg, batch_key=hvg_batch)

    hv = hvgs_df.highly_variable

    hv.to_hdf("hv_genes.h5", key="hvg", mode="w")
    # fmt: off
    hv_idx = hv[hv is True].index
    # fmt: on

    query = census["census_data"][experiment_name].axis_query(
        measurement_name="RNA",
        obs_query=soma.AxisQuery(value_filter=obs_value_filter),
        var_query=soma.AxisQuery(coords=(list(hv_idx),)),
    )

    print("Converting to AnnData")
    ad = query.to_anndata(X_name="raw")

    ad.obs["batch"] = ad.obs["dataset_id"] + ad.obs["assay"] + ad.obs["suspension_type"] + ad.obs["donor_id"]
    ad.var = ad.var.set_index("gene_id", inplace=True)

    print("AnnData conversion completed. Saving...")
    ad.write_h5ad("anndata.h5ad", compression="gzip")
    print("AnnData saved")
