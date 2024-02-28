# Installation

## Requirements

The Census API requires a Linux or MacOS system with:

- Python 3.8 to Python 3.11. Or R, supported versions TBD.
- Recommended: >16 GB of memory.
- Recommended: >5 Mbps internet connection.
- Recommended: for increased performance use the API through a AWS-EC2 instance from the region `us-west-2`. The Census data builds are hosted in a AWS-S3 bucket in that region.

## Python

(Optional) In your working directory, make and activate a virtual environment or conda environment. For example:

```shell
python -m venv ./venv
source ./venv/bin/activate
```

Install the `cellxgene-census` package via pip:

```shell
pip install -U cellxgene-census
```

There are also "experimental" add-on modules that are less stable than the main API, and may have more complex dependencies. To install these,

```shell
pip install -U cellxgene-census[experimental]
```

If installing in a Databricks notebook environment, use `%pip install`. Do not use `%sh pip install`. See the [FAQ](cellxgene_census_docsite_FAQ.md#why-do-i-get-an-error-when-running-import-cellxgene-census-on-databricks).

## R

If installing from **Ubuntu**, you may need to install the following libraries via `apt install`,  `libxml2-dev` `libssl-dev` `libcurl4-openssl-dev`. In addition you must have `cmake` v3.21 or greater.

If installing from **MacOS**, you will need to install the [developer tools `Xcode`](https://apps.apple.com/us/app/xcode/id497799835?mt=12).

**Windows** is not supported.

From an R session, first install `tiledb` from R-Universe, the latest release in CRAN is not yet available.

```r
install.packages(
  "cellxgene.census",
  repos=c('https://chanzuckerberg.r-universe.dev', 'https://cloud.r-project.org')
)
```

To be able to export Census data to `Seurat` or `SingleCellExperiment` you also need to install their respective packages.

```r
# Seurat
install.packages("Seurat")

# SingleCellExperiment
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("SingleCellExperiment")
```
