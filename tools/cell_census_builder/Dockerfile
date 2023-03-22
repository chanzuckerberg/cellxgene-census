FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG COMMIT_SHA
ENV COMMIT_SHA=${COMMIT_SHA}

RUN apt update && apt -y full-upgrade && apt -y install python3.10-venv python3-pip awscli

ADD entrypoint.sh /
ADD dist/ /tools/cell_census_builder

RUN python3 -m pip install -U pip
RUN python3 -m pip -v install /tools/cell_census_builder/*.whl

WORKDIR /census-build

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
