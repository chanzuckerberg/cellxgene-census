# Installation 

## Requirements

The Census API requires a Linux or MacOS system with:

- Python 3.7 to Python 3.10. Or R, supported versions TBD.
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

If installing in a Databricks notebook environment, use `%pip install`. Do not use `%sh pip install`. See the [FAQ](cellxgene_census_docsite_FAQ.md#why-do-i-get-an-error-when-running-import-cellxgene_census-on-databricks).

## R

The R package will be soon deposited into R-Universe. In the meantime you can directly install from github using the [devtools](https://devtools.r-lib.org/) R package.

From an R session:

```r
install.packages("devtools")
devtools::install_github("chanzuckerberg/cellxgene-census/api/r/cellxgene.census")
```
