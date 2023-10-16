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

    adata_config = config["anndata"]
    batch_key = adata_config.get("batch_key")
    ad_filename = adata_config.get("model_filename")

    # TODO: ensure that the anndata we're loading doesn't have obs filtering

    model_config = config.get("model")
    model_filename = model_config.get("filename")
    n_latent = model_config.get("n_latent")

    ad = anndata.read_h5ad(ad_filename)
    model = scvi.model.SCVI.load(model_filename, adata=ad)

    idx = ad.obs.index.to_numpy()

    latent = model.get_latent_representation(ad)

    del model, ad
    gc.collect()

    # i, j = np.meshgrid(idx, range(n_latent), indexing='ij')
    #triplets = np.column_stack(ar.ravel() for ar in (i, j, latent))
    #np.savetxt("latent_triplets.csv", triplets, delimiter=",", fmt='%s,%i,%.9f')

    with open('latent-idx.npy', 'wb') as f:
        np.save(f, idx)

    with open('latent.npy', 'wb') as f:
        np.save(f, latent)

    # np.savetxt("latent.csv", latent, delimiter=",")
    # np.save("latent.npy", latent)