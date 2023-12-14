# Builds a docker image with:
# - PyTorch+CUDA
# - Geneformer
# - cellxgene_census
# - our Census-Geneformer training scripts
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set the tiledbsoma version used to write the embeddings SparseNDArray, to ensure
# compatibility with the Census embeddings curator
ARG EMBEDDINGS_TILEDBSOMA_VERSION=1.4.4
ARG GENEFORMER_VERSION=8df5dc1

RUN apt update && apt install -y python3-venv git-lfs pigz
RUN git lfs install
ENV GIT_SSL_NO_VERIFY=true
RUN pip install \
        transformers[torch] \
        "cellxgene_census[experimental] @ git+https://github.com/chanzuckerberg/cellxgene-census.git#subdirectory=api/python/cellxgene_census" \
        git+https://huggingface.co/ctheodoris/Geneformer@${GENEFORMER_VERSION}
RUN pip install owlready2 boto3

# workaround for unknown problem blocking `import geneformer`:
#   https://github.com/microsoft/TaskMatrix/issues/116#issuecomment-1565431850
RUN pip uninstall -y transformer-engine
# smoke test
RUN python3 -c 'import geneformer; import cellxgene_census; cellxgene_census.open_soma()'

RUN mkdir /census-geneformer
WORKDIR /census-geneformer
# clone Geneformer separately to get LFS files
RUN git clone --recursive https://huggingface.co/ctheodoris/Geneformer \
        && git -C Geneformer checkout ${GENEFORMER_VERSION}

# prepare a venv with tiledbsoma ${EMBEDDINGS_TILEDBSOMA_VERSION}
RUN python3 -m venv --system-site-packages embeddings_tiledbsoma_venv && \
    . embeddings_tiledbsoma_venv/bin/activate && \
    pip install tiledbsoma==${EMBEDDINGS_TILEDBSOMA_VERSION}

COPY *.py .
COPY helpers ./helpers
COPY finetune-geneformer.config.yml .
