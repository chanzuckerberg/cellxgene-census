import cellxgene_census
import tiledbsoma as soma
import torch
import yaml
from cellxgene_census.experimental.pp import highly_variable_genes

import scvi

file = "scvi-config.yaml"

with open(file) as f:
    config = yaml.safe_load(f)

census = cellxgene_census.open_soma(census_version="latest")

census_config = config.get("census")
experiment_name = census_config.get("organism")
obs_value_filter = census_config.get("obs_query")

query = census["census_data"][experiment_name].axis_query(
    measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
)

hvg_config = config.get("hvg")
top_n_hvg = hvg_config.get("top_n_hvg")
hvg_batch = hvg_config.get("hvg_batch")
min_genes = hvg_config.get("min_genes")

hvgs_df = highly_variable_genes(query, n_top_genes=top_n_hvg, batch_key=hvg_batch)

hv = hvgs_df.highly_variable
hv_idx = hv[hv == True].index  # fmt: skip

query = census["census_data"][experiment_name].axis_query(
    measurement_name="RNA",
    obs_query=soma.AxisQuery(value_filter=obs_value_filter),
    var_query=soma.AxisQuery(coords=(list(hv_idx),)),
)
ad = query.to_anndata(X_name="raw")

ad.obs["batch"] = ad.obs["dataset_id"] + ad.obs["assay"] + ad.obs["suspension_type"] + ad.obs["donor_id"]

ad.write_h5ad("anndata.h5ad", compression="gzip")

# scvi settings
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

scvi.model.SCVI.setup_anndata(ad, batch_key="batch")

model_config = config.get("model")
n_hidden = model_config.get("n_hidden")
n_latent = model_config.get("n_latent")
n_layers = model_config.get("n_layers")
dropout_rate = model_config.get("dropout_rate")

model = scvi.model.SCVI(ad, n_layers=n_layers, n_latent=n_latent, gene_likelihood="nb")

train_config = config.get("train")
max_epochs = train_config.get("max_epochs")
batch_size = train_config.get("batch_size")
train_size = train_config.get("train_size")
early_stopping = train_config.get("early_stopping")
devices = train_config.get("devices")

trainer_config = train_config.get("trainer")

training_plan_config = config.get("training_plan")

scvi.settings.dl_num_workers = train_config.get("num_workers")

model.train(
    max_epochs=max_epochs,
    batch_size=batch_size,
    train_size=train_size,
    early_stopping=early_stopping,
    plan_kwargs=training_plan_config,
    strategy="ddp_find_unused_parameters_true",  # Required for Multi-GPU training.
    devices=devices,
    **trainer_config,
)

torch.save(model.module.state_dict(), "scvi.model")
