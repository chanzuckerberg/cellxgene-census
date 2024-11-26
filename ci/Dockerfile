FROM ghcr.io/actions/actions-runner:latest
USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get -y upgrade

RUN apt install -y \
  build-essential \
  libxml2-dev \
  libssl-dev \
  libcurl4-openssl-dev \
  cmake \
  curl \
  python3.11 \
  pip \
  python3.11-venv \
  git-all

RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt install -y python3.12   
RUN apt-get update

USER runner