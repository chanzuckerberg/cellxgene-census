# ReadMe

Demonstration notebooks for the CZ CELLxGENE Discover Census. There are two kinds of notebooks:

1. **API mechanics** — Located under `api_demo` these notebooks provide technical demonstrations of the Census API capabilities.
2. **Computational biology analysis** — Located under `analysis_demo` these notebooks provide an overview of the data in the Census, how to access it and how to use the it in an analytical framework.

## Dependencies

You must be on a Linux or MacOS system, with the following installed:

* Python 3.8 to 3.11
* Jupyter or some other means of running notebooks (e.g., vscode)

For now, it is recommended that you do all this on a host with sufficient memory,
and a high bandwidth connection to AWS S3 in the us-west-2 region, e.g., an m6i.8xlarge.
If you utilize AWS, Ubuntu 20 or 22 AMI are recommended (AWS AMI should work fine, but has
not been tested).

I also recommend you use a `d` instance type, and mount all of the NVME drives as swap,
as it will keep you from running out of RAM.

## Set up Python environment

1. (optional, but highly recommended) In your working directory, make and activate a virtual environment. For example:

    ```shell
      python -m venv ./venv
      source ./venv/bin/activate
    ```

2. Install the required dependencies:

    ```shell
      pip install -U -r cellxgene-census/api/python/notebooks/requirements.txt
    ```

## Verify your installation

Check that your installation works - this make take a few seconds, as it loads metadata from S3:

```shell
$ python -c 'import cellxgene_census; print(cellxgene_census.open_soma().soma_type)'
SOMACollection
```

## Run notebooks

Run notebooks, which you can find in the `cellxgene-census/api/python/notebooks` directory.

## For more help

If you have difficulties or questions, feel to reach out to us using Github issues, or any of the other means described in the [README](../../../README.md).
