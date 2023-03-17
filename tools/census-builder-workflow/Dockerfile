FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG COMMIT_SHA
ENV COMMIT_SHA=${COMMIT_SHA}

RUN apt update && apt -y install python3.10-venv python3-pip awscli

ADD cell_census_builder/ /tools/cell_census_builder
ADD scripts/requirements.txt .
ADD entrypoint.py .
ADD build-census.yaml .

RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["./entrypoint.py"]
