FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG COMMIT_SHA
ENV COMMIT_SHA=${COMMIT_SHA}


# RUN apt-get update && \
#     apt-get install -y python3 libhdf5-dev python3-h5py gettext moreutils build-essential libxml2-dev python3-dev python3-pip zlib1g-dev python3-requests python3-aiohttp llvm jq && \
#     rm -rf /var/lib/apt/lists/*

RUN apt update && apt -y install python3.10-venv python3-pip awscli gh jq

ADD tools/cell_census_builder/ /tools/cell_census_builder
ADD tools/scripts/requirements.txt .
ADD entrypoint.py .
ADD build-census.yaml .

RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["./entrypoint.py"]
