# Installing the Census APIs 

## Requirements

The Census API requires a Linux or MacOS system with:

- Python 3.7 to Python 3.10. Or R, supported versions TBD.
- Recommended: >16 GB of memory.
- Recommended: >5 Mbps internet connection. 
- Recommended: for increased performance use the API through a AWS-EC2 instance from the region `us-west-2`. The Census data builds are hosted in a AWS-S3 bucket in that region.


## Python

1. (Optional) In your working directory, make and activate a virtual environment or conda environment. For example:

```shell
python -m venv ./venv
source ./venv/bin/activate
```

2. Install the `cellxgene-census` package via pip:

```shell
pip install -U cellxgene-census
```

## R

The R package will be soon deposited in R-Universe. In the meantime you directly install from github using the [devtools](https://devtools.r-lib.org/) R package.

From an R session:

```r
install.packages("devtools")
devtools::install_github("chanzuckerberg/cellxgene-census/api/r/cellxgene.census")
```