FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG COMMIT_SHA
ENV COMMIT_SHA=${COMMIT_SHA}

# Ubuntu 22 contains only the python3.11 RC as of 2023-12-21, so use deadsnakes
RUN apt update && \
  apt install -y software-properties-common && \
  add-apt-repository -y ppa:deadsnakes/ppa && \
  apt update && \
  apt -y full-upgrade && \
  apt -y install python3.11 python3.11-venv python3-pip awscli && \
  apt-get clean

# set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
  update-alternatives --set python3 /usr/bin/python3.11

ADD entrypoint.sh /
ADD dist/ /tools/cellxgene_census_builder

RUN python3.11 -m pip install -U pip Cython wheel build
RUN python3.11 -m pip install /tools/cellxgene_census_builder/*.whl

WORKDIR /census-build

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
