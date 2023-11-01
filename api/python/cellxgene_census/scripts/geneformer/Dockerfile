# Builds a docker image with:
# - PyTorch+CUDA
# - Geneformer
# - cellxgene_census
# - our Census-Geneformer training scripts
FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt update && apt install -y git-lfs pigz
RUN git lfs install
RUN pip install \
        transformers[torch] \
        "cellxgene_census[experimental] @ git+https://github.com/chanzuckerberg/cellxgene-census.git#subdirectory=api/python/cellxgene_census" \
        git+https://huggingface.co/ctheodoris/Geneformer
RUN pip install owlready2

# workaround for unknown problem blocking `import geneformer`:
#   https://github.com/microsoft/TaskMatrix/issues/116#issuecomment-1565431850
RUN pip uninstall -y transformer-engine
# smoke test
RUN python3 -c 'import geneformer; import cellxgene_census; cellxgene_census.open_soma()'

RUN mkdir /census-geneformer
WORKDIR /census-geneformer
# clone Geneformer separately to get LFS files
RUN git clone --recursive https://huggingface.co/ctheodoris/Geneformer
COPY *.py .
COPY helpers ./helpers
COPY finetune-geneformer.config.yml .
