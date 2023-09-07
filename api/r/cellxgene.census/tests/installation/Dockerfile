FROM ubuntu:22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  automake \
  build-essential \
  cmake \
  git \
  libcurl4-openssl-dev \
  libssl-dev \
  libxml2-dev \
  r-base \
  zlib1g-dev
ADD install.R /tmp
RUN Rscript /tmp/install.R
