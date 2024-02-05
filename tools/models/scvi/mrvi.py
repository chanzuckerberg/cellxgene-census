import scvi_v2

print(scvi_v2.__file__)


import torch

torch.manual_seed(0)

import anndata as ad
import flax.linen as nn
import yaml
from lightning.pytorch.loggers import TensorBoardLogger

file = "mrvi-config.yaml"

if __name__ == "__main__":
    with open(file) as f:
        config = yaml.safe_load(f)

    adata_config = config["anndata"]
    filename = adata_config.get("model_filename")

    scvi_dataset = ad.read_h5ad(filename)

    train_kwargs = {
        "early_stopping": True,
    }

    plan_kwargs = {"lr": 1e-3, "n_epochs_kl_warmup": 20}

    model_kwargs = {
        "n_latent": 100,
        "n_latent_u": 20,
        "qz_nn_flavor": "attention",
        "px_nn_flavor": "attention",
        "qz_kwargs": {"use_map": False, "stop_gradients": False, "stop_gradients_mlp": True, "dropout_rate": 0.03},
        "px_kwargs": {
            "stop_gradients": False,
            "stop_gradients_mlp": True,
            "h_activation": nn.softmax,
            "dropout_rate": 0.03,
            "low_dim_batch": True,
        },
        "learn_z_u_prior_scale": False,
        "z_u_prior": False,
        "u_prior_mixture": True,
        "u_prior_mixture_k": 100,
    }

    scvi_dataset.obs["nuisance"] = (
        # scvi_dataset.obs['dataset_id'].astype(str) + '_' +
        scvi_dataset.obs["assay"].astype(str)
        + "_"
        + scvi_dataset.obs["suspension_type"].astype(str)
    )
    scvi_dataset.obs["sample"] = (
        scvi_dataset.obs["dataset_id"].astype(str) + "_" + scvi_dataset.obs["donor_id"].astype(str)
    )

    model_config = config.get("model")
    n_hidden = model_config.get("n_hidden")
    n_latent = model_config.get("n_latent")
    n_layers = model_config.get("n_layers")
    dropout_rate = model_config.get("dropout_rate")
    output_filename = model_config.get("filename")

    scvi_v2.MrVI.setup_anndata(scvi_dataset, sample_key="sample", batch_key="nuisance", labels_key="cell_type")
    mrvi_model = scvi_v2.MrVI(scvi_dataset, **model_kwargs)

    logger = TensorBoardLogger("mrvi_tb_logs", name="mrvi_50_epochs")

    mrvi_model.train(max_epochs=50, batch_size=4096, plan_kwargs=plan_kwargs, **train_kwargs)

    mrvi_model.save(output_filename)

    # # Get z representation
    # adata.obsm["X_mrvi_z"] = mrvi_model.get_latent_representation(give_z=True)
    # # Get u representation
    # adata.obsm["X_mrvi_u"] = mrvi_model.get_latent_representation(give_z=False)
    # sc.pp.neighbors(adata, use_rep="X_mrvi_u", key_added="neighbors_mrvi", method='rapids', n_neighbors=30)
    # sc.tl.umap(adata, neighbors_key="neighbors_mrvi", method='rapids')
    # sc.pl.umap(adata, color=['dataset_id', 'cell_subclass', 'suspension_type', 'sex'], ncols=1, frameon=False, wspace=0.4, title='mrVI (Census)')
