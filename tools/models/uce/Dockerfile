# Builds a docker image with:
# - PyTorch+CUDA
# - cellxgene_census
# - our Census-UCE training scripts
#
# Make sure to install nvidia drivers AND nvidia-container-runtime-hook
# To test docker image and gpu usage run:
# docker run --gpus all uce /bin/bash -c "cd ./UCE/ && python3 eval_single_anndata.py --adata_path ./small_census_anndata.h5ad"

FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN mkdir /census-uce
WORKDIR /census-uce

# Set the tiledbsoma version used to write the embeddings SparseNDArray, to ensure
# compatibility with the Census embeddings curator
ARG EMBEDDINGS_TILEDBSOMA_VERSION=1.4.4

RUN pip install cellxgene_census
RUN pip install accelerate

# Copy UCE files, if this is a fresh clone of the census make sure to run:
COPY UCE/ ./UCE/
COPY small_census_anndata.h5ad ./UCE/small_census_anndata.h5ad

# UCE smoke test and download files for UCE 
RUN cd ./UCE/ && python3 eval_single_anndata.py --adata_path ./small_census_anndata.h5ad

RUN apt update && apt install -y python3-venv 
RUN pip install owlready2 boto3

## smoke test
RUN python3 -c 'import cellxgene_census; cellxgene_census.open_soma()'

## prepare a venv with tiledbsoma ${EMBEDDINGS_TILEDBSOMA_VERSION}
RUN python3 -m venv --system-site-packages embeddings_tiledbsoma_venv && \
    . embeddings_tiledbsoma_venv/bin/activate && \
    pip install tiledbsoma==${EMBEDDINGS_TILEDBSOMA_VERSION}

COPY *.py .
