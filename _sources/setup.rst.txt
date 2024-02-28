Installation
=====

Dependencies
----

You must be on a Linux or MacOS system, with the following installed:

- Python 3.8 to 3.11
- Jupyter or some other means of running notebooks (e.g., vscode)

For now, it is recommended that you do all this on a host with sufficient memory,
and a high bandwidth connection to AWS S3 in the us-west-2 region, e.g., an m6i.8xlarge.
If you utilize AWS, Ubuntu 20 or 22 AMI are recommended (AWS AMI should work fine, but has
not been tested).

I also recommend you use a ``d`` instance type, and mount all of the NVME drives as swap,
as it will keep you from running out of RAM.


Set up Python environment
----

1. (optional, but highly recommended) In your working directory, make and activate a virtual environment. For example: 
::

  $ python -m venv ./venv
  $ source ./venv/bin/activate

2. Install the ``cellxgene_census`` package using ``pip``:
::

  $ pip install -U cellxgene-census

3. Install other third-party packages needed to run the notebooks:
::

  $ pip install scikit-misc scvi-tools


Verify your installation
----

Check that your installation works - this make take a few seconds, as it loads metadata from S3:
::

  $ python -c 'import cellxgene_census; print(cellxgene_census.open_soma().soma_type)'
  SOMACollection

Latest development version
----

If you want to work with the latest development version of cellxgene-census, you can simply clone the repository 
and, from the root directory, install locally via pip:
::

  $ git clone https://github.com/chanzuckerberg/cellxgene-census.git
  $ cd cellxgene-census
  $ pip install -e api/python/cellxgene_census/