FROM ubuntu:22.04

# TILEDB_PY_VERSION should be set such that the TileDB Embedded version will match that used by
# tiledbsoma in cellxgene_census_builder and census_contrib.
# https://github.com/single-cell-data/TileDB-SOMA/blob/1.15.3/libtiledbsoma/cmake/Modules/FindTileDB_EP.cmake#L93 (2.27.0)
#  ==
# https://github.com/TileDB-Inc/TileDB-Py/blob/0.33.3/CMakeLists.txt#L49 (2.27.0)
ARG TILEDB_PY_VERSION=0.33.3
# TILEDB_VECTOR_SEARCH_VERSION should be the newest compatible with TILEDB_PY_VERSION.
# https://github.com/TileDB-Inc/TileDB-Vector-Search/blob/0.11.0/pyproject.toml#L23 (tiledb-py>=0.32.0)
ARG TILEDB_VECTOR_SEARCH_VERSION=0.11.0

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip
RUN pip3 install \
    tiledb==$TILEDB_PY_VERSION \
    tiledb-vector-search==$TILEDB_VECTOR_SEARCH_VERSION

# FIXME: monkey patch tiledb-vector-search 0.11 for https://github.com/TileDB-Inc/TileDB-Vector-Search/issues/564
# This should be removed when we update to a new version addressing that issue.
ADD ingestion.py.patch /tmp
RUN patch /usr/local/lib/python3.10/dist-packages/tiledb/vector_search/ingestion.py /tmp/ingestion.py.patch
