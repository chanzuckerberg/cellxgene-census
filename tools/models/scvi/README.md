# Census trained scVI model

## Training Pipeline

This directory contains a set of scripts that can be used to train [scVI](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html) on the whole Census data, and to generate its latent space representation embeddings.

The model can be trained separately on each experiment (`homo_sapiens` and `mus_musculus`), and produces separate artifacts.

In order to run the training pipeline, three separate files are provided:

[scvi-prepare.py](scvi-prepare.py)

This file prepares an AnnData file that can be fed to the scVI trainer directly. The preparation strategy is:

1. Take all the primary cells from the census with a gene count greater or equal than 300 (`is_primary_data == True and nnz >= 300`).
1. Extract the top 8000 highly variable genes (using the Census `highly_variable_genes` function). Those are serialized to a `hv_genes.pkl` numpy.ndarray.
1. A batch_key column is created by concatenating the `[dataset_id, assay, suspension_type, donor_id]` covariates

The output of this file is an `anndata_model.h5ad` file.

[scvi-train.py](scvi-train.py)

This file takes the AnnData file from the previous step and trains an scVI model on it. See [scvi-config.yaml](scvi.config.yaml) for an up-to-date list of model and training parameters.

The resulting model weights are saved to an `scvi.model` directory. Tensorboard logs are also available as part of the output.

[scvi-create-latent-update.py](scvi-create-latent-update.py)

This file takes the previously generated model and obtains the latent space representation to generate cell embeddings. The generation strategy is:

1. Take all the cells from the Census (since we want to generate embeddings for every cell)
1. Take the same highly variable genes from the `prepare` step
1. Generate an AnnData with the same properties as the `prepare` step
1. Call the `scvi.model.SCVI.load_query_data()` function on this AnnData. This allows to work on a dataset that has more cells than the one the model is trained on (which is required so that the model doesn't need to be re-trained from scratch on each Census version). A further pass of training is possible, but we just set `is_trained = True` to skip it.
1. We call `get_latent_representation()` to generate the embeddings
1. Both the final h5ad file, the embeddings and the cell index are saved as part of the output.

## Selection of model parameters

The final selection of parameters for the training phase was based on a hyper parameter search as described in the [CELLxGENE Discover Census scvi-tools initial autotune report](https://github.com/YosefLab/census-scvi/blob/main/experiments/autotune/notebooks/2023_09_autotune_report.ipynb)

## Environment setup

The training has been performed on an AWS EC2 machine (instance type: g4dn.12xlarge), running on Ubuntu 20.04. Run [scvi-init.sh](scvi-init.sh) to set up the environment required to run the pipeline.
