import scvi_v2

print(scvi_v2.__file__)


import torch

torch.manual_seed(0)

import anndata as ad
import numpy as np
import yaml

file = "mrvi-config.yaml"

if __name__ == "__main__":
    with open(file) as f:
        config = yaml.safe_load(f)

    adata_config = config["anndata"]
    filename = adata_config.get("model_filename")

    census_config = config["census"]
    experiment_name = census_config.get("organism")

    scvi_dataset = ad.read_h5ad(filename)

    scvi_dataset.obs["nuisance"] = (
        # scvi_dataset.obs['dataset_id'].astype(str) + '_' +
        scvi_dataset.obs["assay"].astype(str)
        + "_"
        + scvi_dataset.obs["suspension_type"].astype(str)
    )
    scvi_dataset.obs["sample"] = (
        scvi_dataset.obs["dataset_id"].astype(str) + "_" + scvi_dataset.obs["donor_id"].astype(str)
    )

    # hv = pd.read_pickle("hv_genes.pkl")
    # hv_idx = hv[hv].index

    # census = cellxgene_census.open_soma(census_version="2023-12-15")

    # obs_query = None # not for now

    # query = census["census_data"][experiment_name].axis_query(
    #     measurement_name="RNA",
    #     obs_query=obs_query,
    #     var_query=soma.AxisQuery(coords=(list(hv_idx),)),
    # )

    # idx = query.obs(column_names=["soma_joinid"]).concat().to_pandas().index.to_numpy()

    # adata = query.to_anndata(X_name="raw")

    model_config = config.get("model")
    model_filename = model_config.get("filename")

    # May or may not be necessary
    # scvi_v2.MrVI.setup_anndata(scvi_dataset, sample_key="sample", batch_key="nuisance", labels_key="cell_type")

    mrvi_model = scvi_v2.MrVI.load("mrvi.model", adata=scvi_dataset)

    latent = mrvi_model.get_latent_representation(give_z=False)

    # with open("mrvi-latent-idx.npy", "wb") as f:
    #     np.save(f, idx)

    with open("mrvi-latent.npy", "wb") as f:
        np.save(f, latent)
