# ReadMe

Demonstration notebooks for the CELLxGENE Cell Census

This is a quick start on how to run the notebooks.  It is Linux-flavored.

## Dependencies

You must be on a Linux or MacOS system, with the following installed:
* Python 3.9+
* C++ 17 build tools
* cmake 3.21 or later
* git
* Jupyter or some other means of running notebooks (e.g., vscode)

For now, it is recommended that you do all this on an EC2 instance, and choose one 
with sufficient memory and bandwidth (eg, m6i.16xlarge). If you do, it is recommended
that you use the Ubuntu 20 or 22 AMI (AWS AMI should work fine, but has not been tested).

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
```bash
  $ python -m venv ./venv
  $ source ./venv/bin/activate
```
2. Build and install SOMA into your virtual environment by following the instructions in `TileDB-SOMA/apis/python/README.md`
3. Install the `cell_census` package:
```bash
  $ pip install -e cell-census/api/python/cell_census/
```
4. Install packages needed to run notebooks:
```bash
  $ pip install scikit-misc
```

## Verify your installation
Check that your installation works - this make take a few seconds, as it loads metadata from S3:
```bash
$ python -c 'import cell_census; print(cell_census.open_soma().soma_type)'
SOMACollection
```

## Run notebooks
Run notebooks, which you can find in the `cell-census/api/python/notebooks` directory.
