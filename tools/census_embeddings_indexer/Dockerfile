FROM ubuntu:22.04
# TILEDB_VECTOR_SEARCH_VERSION should be the newest that doesn't need a newer version of tiledb
# than the client tiledbsoma: https://github.com/TileDB-Inc/TileDB-Vector-Search/blob/0.2.2/pyproject.toml
ARG TILEDB_VECTOR_SEARCH_VERSION=0.2.2
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip
RUN pip3 install \
    cellxgene_census \
    tiledb-vector-search==$TILEDB_VECTOR_SEARCH_VERSION
