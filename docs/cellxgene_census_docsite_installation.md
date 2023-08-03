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

If installing in a Databricks notebook environment, use `%pip install`. Do not use `%sh pip install`. See the [FAQ](cellxgene_census_docsite_FAQ.md#why-do-i-get-an-error-when-running-import-cellxgene-census-on-databricks).

## R

From an R session, first install `tiledb` from R-Universe, the latest release in CRAN is not yet available.

```r
install.packages(
  "tiledb",
  version = "0.20.2", 
  repos=c('https://tiledb-inc.r-universe.dev','https://cloud.r-project.org') 
)
```

Then install `cellxgene.census` from R-Universe.

```r
install.packages(
  "cellxgene.census",
  repos=c('https://tiledb-inc.r-universe.dev','https://cloud.r-project.org') 
)
```