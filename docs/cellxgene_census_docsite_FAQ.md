# FAQ

Last updated: Apr, 2023.

- [Why should I use the Census?](#why-should-i-use-the-census)
- [What data is contained in the Census?](#what-data-is-contained-in-the-census)
- [How do I cite the use of the Census for a publication?](#how-do-i-cite-the-use-of-the-census-for-a-publication)
- [Why does the Census not have a normalized layer or embeddings?](#why-does-the-census-not-have-a-normalized-layer-or-embeddings)
- [How does the Census differentiate from other services?](#how-does-the-census-differentiate-from-other-services)
- [Can I query human and mouse data in a single query?](#can-i-query-human-and-mouse-data-in-a-single-query)
- [Where are the Census data hosted?](#where-are-the-census-data-hosted)
- [Can I retrieve the original H5AD datasets from which the Census was built?](#can-i-retrieve-the-original-h5ad-datasets-from-which-the-census-was-built)
- [How can I increase the performance of my queries?](#how-can-i-increase-the-performance-of-my-queries)
- [Can I use conda to install the Census Python API?](#can-i-use-conda-to-install-the-census-python-api)
- [How can I ask for support?](#how-can-i-ask-for-support)
- [How can I ask for new features?](#how-can-i-ask-for-new-features)
- [How can I contribute my data to the Census?](#how-can-i-contribute-my-data-to-the-census)
- [Why do I get an `ArraySchema` error when opening the Census?](#why-do-i-get-an-arrayschema-error-when-opening-the-census)

## Why should I use the Census?

The Census provides efficient low-latency access via Python and R APIs to most single-cell RNA data from [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/). 

To accelerate your computational research, **you should use the Census if you want to**:

- Easily get slices of data from more than 400 single-cell datasets spanning about 50 M cells from >60 K genes from human or mouse.
- Get these data with standardized and harmonized cell and gene metadata.
- Easily load multi-dataset slices into Scanpy or Seurat.
- Implement out-of-core (a.k.a online) operations for larger-than-memory processes.


For example you could easily get "*all T-cells from Lung with COVID-19*" into an [AnnData](https://anndata.readthedocs.io/en/latest/), [Seurat](https://satijalab.org/seurat/), or into memory-sufficient data chunks via [PyArrow](https://arrow.apache.org/docs/python/index.html) or [R Arrow](https://arrow.apache.org/docs/r/). 


**You should not use the Census if you want to:**

- Access non-standardized cell metadata and gene metadata available in the original [datasets](https://cellxgene.cziscience.com/datasets).
- Access the author-contributed normalized expression values or embeddings. 
- Access all data from a single dataset.
- Access non-RNA or spatial data present in CZ CELLxGENE Discover as it is not yet supported in the Census. 

For all of these cases you should perform web downloads from the [CZ CELLxGENE Discover site](https://cellxgene.cziscience.com/datasets), you can find instructions to do so [here](https://cellxgene.cziscience.com/docs/03__Download%20Published%20Data). 

## What data is contained in the Census?

Most RNA non-spatial data from [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/) is included. You can see a general description of these data and their organization in the [schema description](cellxgene_census_docsite_schema.md) or you can use the APIs to explore the data as indicated in this [tutorial](notebooks/analysis_demo/comp_bio_census_info.ipynb). 

## How do I cite the use of the Census for a publication?

Please follow the [citation guidelines](https://cellxgene.cziscience.com/docs/08__Cite%20cellxgene%20in%20your%20publications) offered by CZ CELLxGENE Discover.


## Why does the Census not have a normalized layer or embeddings?

The Census does not have normalized counts or embeddings because:

- The original normalized values and embeddings are not harmonized or integrated across datasets and are therefore numerically incompatible.
- We have not implemented a general-purpose normalization or embedding generation method to be used across all Census data.

If you have any suggestions for methods that our team should explore please share them with us via a [feature request in the github repository](https://github.com/chanzuckerberg/cellxgene-census/issues/new?assignees=&labels=user+request&template=feature-request.md&title=).

## How does the Census differentiate from other services?

The Census differentiates from existing single-cell services by providing access to the largest corpus of standardized single-cell data via [TileDB-SOMA](https://github.com/single-cell-data/TileDB-SOMA/issues/new/choose). 

Thus, single-cell data from about 50 M cells across >60 K genes, with 11 standardized cell metadata variables and harmonized GENCODE annotations is at your finger tips to:

- Open and read data at low latency from the cloud.
- Query and access data using metadata filters.
- Load and create AnnData objects.
- Load and create Seurat objects.
- From Python create PyArrow objects, SciPy sparse matrices, NumPy arrays, and Pandas data frames.
- From R create R Arrow objects, sparse matrices (via the Matrix package), and standard data frames and (dense) matrices.

## Can I query human and mouse data in a single query?

It is not possible to query both mouse and human data in a single query. This is due to the data from these organisms using different [organism-specific gene annotations](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#required-gene-annotations).

## Where are the Census data hosted?

The Census data is publicly hosted free-of-cost in an Amazon Web Services (AWS) S3 bucket in the [`us-west-2` region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions).

## Can I retrieve the original H5AD datasets from which the Census was built?

Yes, you can use the API function `download_source_h5ad` to do so. For usage see the reference documentation at the [doc-site](https://cellxgene-census.readthedocs.io/en/) or directly from Python or R:

Python

```python
import cellxgene_census
help(cellxgene_census.download_source_h5ad)
```

R

```r
library(cellxgene.census)
?download_source_h5ad
```

## How can I increase the performance of my queries?

Since the access patterns are via the internet, usually the main limiting step for data queries is bandwidth and client location.

We recommend the following to increase query efficency: 

- Utilize a computer connected to high-speed internet.
- Utilize an ethernet connection and not a wifi connection.
- If possible utilize online computing located in the west coast of the US.
   - Highly recommended: [EC2 AWS instances](https://aws.amazon.com/ec2/) in the `us-west-2` region.

## Can I use conda to install the Census Python API?

There is not a conda package available for `cellxgene-census`. However you can use conda in combination with `pip` to install the package in a conda environment:

```bash
conda create -n census_env python=3.10
conda activate census_env
pip install cellxgene-census
```

## How can I ask for support?

You can either submit a [github issue](https://github.com/chanzuckerberg/cellxgene-census/issues/new/choose) or post in the slack channel `#cellxgene-census-users` at the [CZI Slack community](https://cziscience.slack.com/join/shared_invite/zt-czl1kp2v-sgGpY4RxO3bPYmFg2XlbZA#/shared-invite/email).

## How can I ask for new features?

You can submit a [feature request in the github repository](https://github.com/chanzuckerberg/cellxgene-census/issues/new?assignees=&labels=user+request&template=feature-request.md&title=).

## How can I contribute my data to the Census?

To inquire about submitting your data to CZ CELLxGENE Discover you need to follow these [instructions](https://cellxgene.cziscience.com/docs/032__Contribute%20and%20Publish%20Data). 

If you data request is accepted, upon submission the data will automatically get included in the Census if it meets the [biological criteria defined in the Census schema](https://github.com/chanzuckerberg/cellxgene-census/blob/main/docs/cellxgene_census_schema.md#data-included). 

## Why do I get an `ArraySchema` error when opening the Census?

You may get this error if you are trying to open a Census data build with an old version of the Census API. Please update your Python or R Census package.

If the error persists please file a [github issue](https://github.com/chanzuckerberg/cellxgene-census/issues/new/choose).

