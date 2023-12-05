#!/bin/bash

# Can be used to bootstrap a g4dn.* instance with scvi-tools and cellxgene-census

export DEBIAN_FRONTEND=noninteractive

sudo apt -y update 
sudo apt -y install python3-pip
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt -y update 
sudo apt -y install python3.11
sudo apt -y install python3.11-venv
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 3
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
sudo update-alternatives --config python
sudo update-alternatives --config python3
sudo cp /usr/lib/python3/dist-packages/apt_pkg.cpython-38-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/apt_pkg.so

sudo apt -y install libnvidia-gl-535 libnvidia-common-535 libnvidia-compute-535 libnvidia-encode-535 libnvidia-decode-535 nvidia-compute-utils-535  libnvidia-fbc1-535 nvidia-driver-535

ipip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pathlib torch click ray hyperopt
pip install git+https://github.com/scverse/scvi-tools.git


# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install nvidia-cusolver-cu11

pip install scikit-misc
pip install git+https://github.com/chanzuckerberg/cellxgene-census#subdirectory=api/python/cellxgene_census