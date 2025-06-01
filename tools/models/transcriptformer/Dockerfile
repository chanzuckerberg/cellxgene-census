FROM public.ecr.aws/ubuntu/ubuntu:24.04_stable
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git-core git-lfs
RUN pip3 install --break-system-packages \
        transcriptformer==0.3.0 \
        cellxgene-census
WORKDIR /census-transcriptformer
RUN transcriptformer download tf-sapiens
RUN transcriptformer download tf-exemplar
RUN transcriptformer download tf-metazoa

# Set the tiledbsoma version used to write the embeddings SparseNDArray, to ensure
# compatibility with the Census embeddings curator
ARG EMBEDDINGS_TILEDBSOMA_VERSION=1.11.4
# prepare a venv with pinned tiledbsoma ${EMBEDDINGS_TILEDBSOMA_VERSION}, which our embeddings
# generation step will use to output a TileDB array compatible with the Census embeddings curator.
RUN python3 -m venv embeddings_tiledbsoma_venv
RUN embeddings_tiledbsoma_venv/bin/pip install tiledbsoma==${EMBEDDINGS_TILEDBSOMA_VERSION} anndata

COPY planner.py inference_driver.py put_embeddings.py ./
