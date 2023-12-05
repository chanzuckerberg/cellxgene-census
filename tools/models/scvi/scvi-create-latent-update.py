import functools
import gc

import cellxgene_census
import numpy as np
import pandas as pd
import tiledbsoma as soma
import yaml

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
    hv_idx = hv[hv].index
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
    ad.obs["batch"] = functools.reduce(lambda a, b: a + b, [ad.obs[c].astype(str) for c in batch_key])

    ad.var.set_index("feature_id", inplace=True)

    idx = query.obs(column_names=["soma_joinid"]).concat().to_pandas().index.to_numpy()

    del census, query, hv, hv_idx
    gc.collect()

    # TODO: ensure that the anndata we're loading doesn't have obs filtering

    model_config = config.get("model")
    model_filename = model_config.get("filename")
    n_latent = model_config.get("n_latent")

    scvi.model.SCVI.prepare_query_anndata(ad, model_filename)

    vae_q = scvi.model.SCVI.load_query_data(
        ad,
        model_filename,
    )
    # vae_q.train(max_epochs=1, plan_kwargs=dict(weight_decay=0.0))
    vae_q.is_trained = True
    latent = vae_q.get_latent_representation()

    ad.write_h5ad("anndata-full.h5ad", compression="gzip")

    del vae_q, ad
    gc.collect()

    with open("latent-idx.npy", "wb") as f:
        np.save(f, idx)

    with open("latent.npy", "wb") as f:
        np.save(f, latent)
