# Builds a docker image with:
# - CUDA+PyTorch
# - Geneformer
# - cellxgene_census
# - our Census-Geneformer training scripts
FROM nvcr.io/nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set the tiledbsoma version used to write the embeddings SparseNDArray, to ensure
# compatibility with the Census embeddings curator
ARG EMBEDDINGS_TILEDBSOMA_VERSION=1.9.5
ARG CELLXGENE_CENSUS_VERSION=main
ARG GENEFORMER_VERSION=471eefc

RUN apt update && apt install -y build-essential python3-pip python3-venv git-lfs pigz libcurl4-openssl-dev
RUN git lfs install

ENV GIT_SSL_NO_VERIFY=true
RUN pip install --upgrade pip setuptools setuptools_scm
RUN pip install torch torchdata --index-url https://download.pytorch.org/whl/cu118
                                                                             # ^^^ match the base image CUDA version!
RUN pip install owlready2 boto3 transformers[torch]
# workaround for unknown problem blocking `import geneformer`:
#   https://github.com/microsoft/TaskMatrix/issues/116#issuecomment-1565431850
RUN pip uninstall -y transformer-engine

RUN mkdir /census-geneformer
WORKDIR /census-geneformer
RUN git clone https://github.com/chanzuckerberg/cellxgene-census.git \
        && git -C cellxgene-census checkout ${CELLXGENE_CENSUS_VERSION}
RUN pip install cellxgene-census/api/python/cellxgene_census
RUN git clone --recursive https://huggingface.co/ctheodoris/Geneformer \
        && git -C Geneformer checkout ${GENEFORMER_VERSION}
RUN pip install -e Geneformer

# smoke test
RUN python3 -c 'import geneformer; import cellxgene_census; from cellxgene_census.experimental.ml.huggingface import GeneformerTokenizer; cellxgene_census.open_soma()'

# prepare a venv with pinned tiledbsoma ${EMBEDDINGS_TILEDBSOMA_VERSION}, which our embeddings
# generation step will use to output a TileDB array compatible with the Census embeddings curator.
RUN python3 -m venv --system-site-packages embeddings_tiledbsoma_venv && \
    . embeddings_tiledbsoma_venv/bin/activate && \
    pip install tiledbsoma==${EMBEDDINGS_TILEDBSOMA_VERSION}

COPY gf-95m ./gf-95m/
COPY helpers ./helpers
COPY *.py ./
COPY finetune-geneformer.config.yml .
