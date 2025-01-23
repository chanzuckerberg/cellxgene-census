FROM ubuntu:22.04

# TILEDB_PY_VERSION should match the version required (via tiledbsoma) by the Census builder and
# embeddings curator tool.
# https://github.com/chanzuckerberg/cellxgene-census/blob/v1.16.2/tools/cellxgene_census_builder/pyproject.toml#L37 (tiledbsoma 1.11.4)
# https://github.com/single-cell-data/TileDB-SOMA/blob/1.11.4/apis/python/setup.py#L345 (tiledb-py 0.29)
ARG TILEDB_PY_VERSION=0.29.1
# TILEDB_VECTOR_SEARCH_VERSION should be the newest compatible with TILEDB_PY_VERSION.
# https://github.com/TileDB-Inc/TileDB-Vector-Search/blob/0.4.1/pyproject.toml (tiledb-py 0.29)
ARG TILEDB_VECTOR_SEARCH_VERSION=0.4.1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip
RUN pip3 install \
    tiledb==$TILEDB_PY_VERSION \
    tiledb-vector-search==$TILEDB_VECTOR_SEARCH_VERSION
