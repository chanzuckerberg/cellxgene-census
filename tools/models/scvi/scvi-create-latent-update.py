import functools
import gc

import cellxgene_census
import numpy as np
import pandas as pd
import scvi
import tiledbsoma as soma
import yaml

file = "scvi-config.yaml"

if __name__ == "__main__":
    with open(file) as f:
        config = yaml.safe_load(f)

    census = cellxgene_census.open_soma(census_version="2023-12-15")

    census_config = config.get("census")
    experiment_name = census_config.get("organism")
    obs_value_filter = census_config.get("obs_query")

    hv = pd.read_pickle("hv_genes.pkl")
    hv_idx = hv[hv].index

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
    adata = query.to_anndata(X_name="raw")
    adata.obs["batch"] = functools.reduce(lambda a, b: a + b, [adata.obs[c].astype(str) for c in batch_key])

    adata.var.set_index("feature_id", inplace=True)

    idx = query.obs(column_names=["soma_joinid"]).concat().to_pandas().index.to_numpy()

    del census, query, hv, hv_idx
    gc.collect()

    model_config = config.get("model")
    model_filename = model_config.get("filename")
    n_latent = model_config.get("n_latent")

    scvi.model.SCVI.prepare_query_anndata(adata, model_filename)

    vae_q = scvi.model.SCVI.load_query_data(
        adata,
        model_filename,
    )
    vae_q.is_trained = True
    qz_m, qz_v = vae_q.get_latent_representation(return_dist=True)

    adata.obsm["_scvi_latent_qzm"], adata.obsm["_scvi_latent_qzv"] = qz_m, qz_v
    vae_q.minify_adata(use_latent_qzm_key="_scvi_latent_qzm", use_latent_qzv_key="_scvi_latent_qzv")
    vae_q.save("final_scvi_optimized", save_anndata=True)

    del vae_q, adata, qz_v
    gc.collect()

    with open("latent-idx.npy", "wb") as f:
        np.save(f, idx)

    with open("latent.npy", "wb") as f:
        np.save(f, qz_m)
