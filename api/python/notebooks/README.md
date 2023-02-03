# ReadMe

Demonstration notebooks for the CELLxGENE Cell Census. There are two kinds of notebooks:

1. **API mechanics** — Located under `api_demo` these notebooks provide technical demonstrations of the Cell Census API capabilities.
2. **Computational biology analysis** — Located under `analysis_demo` these notebooks provide an overview of the data in the Cell Census, how to access it and how to use the it in an analytical framework.

## Dependencies

You must be on a Linux or MacOS system, with the following installed:
* Python 3.8+
* C++ 17 build tools
* cmake 3.21 or later
* git
* Jupyter or some other means of running notebooks (e.g., vscode)

For now, it is recommended that you do all this on a host with sufficient memory,
and a high bandwidth connection to AWS S3 in the us-west-2 region, e.g., an m6i.16xlarge.
If you utilize AWS, Ubuntu 20 or 22 AMI are recommended (AWS AMI should work fine, but has
not been tested).

I also recommend you use a `d` instance type, and mount all of the NVME drives as swap,
as it will keep you from running out of RAM.

## Step 1: Clone Repos

On your target host:
1. Make a new working directory and `cd` into it
2. Clone both TileDB-SOMA and soma-scratch.
```bash
$ git clone https://github.com/single-cell-data/TileDB-SOMA.git
$ git clone https://github.com/chanzuckerberg/cell-census.git
```

## Step 2: Set up Python environment
1. In your working directory, make and activate a virtual environment.
```shell
  $ python -m venv ./venv
  $ source ./venv/bin/activate
```
2. Build and install SOMA into your virtual environment by following the instructions in `TileDB-SOMA/apis/python/README.md`
3. Install the `cell_census` package:
```shell
  $ pip install -e cell-census/api/python/cell_census/
```
4. Install packages needed to run notebooks:
```shell
  $ pip install scikit-misc scvi-tools
```

## Verify your installation
Check that your installation works - this make take a few seconds, as it loads metadata from S3:
```shell
$ python -c 'import cell_census; print(cell_census.open_soma().soma_type)'
SOMACollection
```

## Run notebooks
Run notebooks, which you can find in the `cell-census/api/python/notebooks` directory.
