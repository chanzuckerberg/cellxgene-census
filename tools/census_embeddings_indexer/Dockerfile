FROM ubuntu:22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip
RUN pip3 install tiledb-vector-search==0.4.1
RUN pip3 install cellxgene_census
