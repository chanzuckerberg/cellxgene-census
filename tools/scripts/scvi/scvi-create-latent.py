import gc

import anndata
import cellxgene_census
import numpy as np
import pandas as pd
import tiledbsoma as soma
import yaml
import functools

import scvi

file = "scvi-config.yaml"

if __name__ == "__main__":
    with open(file) as f:
        config = yaml.safe_load(f)

    census = cellxgene_census.open_soma(census_version="latest")

    census_config = config.get("census")
    experiment_name = census_config.get("organism")
    obs_value_filter = census_config.get("obs_query")

    hv = pd.read_pickle("hv_genes.pkl")
    # fmt: off
    hv_idx = hv[hv == True].index
    # fmt: on

    if obs_value_filter is not None:
        obs_query = soma.AxisQuery(value_filter=obs_value_filter)
    else:
        obs_query = None

    query = census["census_data"][experiment_name].axis_query(
        measurement_name="RNA",
        obs_query=obs_query,
        var_query=soma.AxisQuery(coords=(list(hv_idx),)),
    )

    adata_config = config["anndata"]
    batch_key = adata_config.get("batch_key")
    ad_filename = adata_config.get("model_filename")

    print("Converting to AnnData")
    ad = query.to_anndata(X_name="raw")
    ad.obs["batch"] = functools.reduce(lambda a, b: a+b, [ad.obs[c].astype(str) for c in batch_key])

    ad.var.set_index("feature_id", inplace=True)

    del census, query, hv, hv_idx
    gc.collect()

    # TODO: ensure that the anndata we're loading doesn't have obs filtering

    model_config = config.get("model")
    model_filename = model_config.get("filename")

    model = scvi.model.SCVI.load(model_filename, adata=ad)

    latent = model.get_latent_representation(ad)

    print(ad.shape)
    print(latent.shape)

    idx = ad.obs.index.to_numpy()
    i, j = np.meshgrid(idx, range(200), indexing='ij')
    triplets = np.column_stack(ar.ravel() for ar in (i, j, latent))
    np.savetxt("latent_triplets.csv", triplets, delimiter=",", fmt='%i %i %.9f')

    # np.savetxt("latent.csv", latent, delimiter=",")
    # np.save("latent.npy", latent)
