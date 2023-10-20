import cellxgene_census
import numpy as np
import tiledbsoma as soma

from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
import anndata
import yaml
import scanpy

n_components = 50

file = "scvi-config.yaml"

if __name__ == "__main__":
    with open(file) as f:
        config = yaml.safe_load(f)

    adata_config = config["anndata"]
    batch_key = adata_config.get("batch_key")
    ad_filename = adata_config.get("model_filename")

    ad = anndata.read_h5ad(ad_filename)

    print(ad)


    pca = scanpy.tl.pca(ad, n_comps=n_components)

    with open('pca.npy', 'wb') as f:
        np.save(f, ad.obsm["X_pca"])

    # print(ad.obsm["X_pca"].shape)


# Run PCA on the original matrix (the same data you input in SCVI. do not use raw counts, but use log transform data. see sc.pp.log1p() in scanpy). how many components? 50. plot the variance of each component

# Run SCVI on the same matrix (raw counts) and get the latent space. The number of latent dimensions does not need match the number of PCA components.

# Once you have both you can run scib metrics