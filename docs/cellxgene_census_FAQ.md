# Frequently Asked Questions for CZ CELLxGENE Discover Census

Last updated: Apr, 2023.

- [Why should I use the Census?](#Why-should-I-use-the-Census)

## Why should I use the Census?

The Census provides efficient low-latency access via Python and R APIs to most single-cell RNA data from [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/). 

To accelerate your computational research, **you should use the Census**:

- If you want easily get slices of data from more than 400 single-cell datasets spanning about 50 mi cells from >60 K genes.
- If you want easy access to a subset of data from a dataset or from multiple datasets.
- If you want to get these data with standardized and harmonized cell and gene metadata.
- If you want to easily load multi-dataset slices into Scanpy or Seurat.
- If you want to implement out-of-core (a.k.a online) operations for larger than  memory processes.


For example you could easily get "all T-cells from Lung with COVID-19" into an [AnnData](https://anndata.readthedocs.io/en/latest/), [Seurat](https://satijalab.org/seurat/), or into memory-sufficient data chunks via [Arrow](https://arrow.apache.org/). 


**You should not use the Census:**

- If you want access to non-standardized cell metadata and gene metadata available in the original [datasets](https://cellxgene.cziscience.com/datasets).
- If you want access to the author-contributed normalized expression vaues.
- If you want access to all data from a single dataset.

## Why does the Census not have a normalized layer or embeddings?

The Census does not have normalized counts or embeddings because:

- The original normalized values and embeddings are not harmonized or integrated across datasets and therefore numerically incompatible.
- We have not chosen nor implemented a general-purpose normalization or embedding generation method to be used across all Census data.

If you have any suggestions for methods that our team should explore please share with us via a [feature request in the github repository](https://github.com/chanzuckerberg/cellxgene-census/issues/new?assignees=&labels=user+request&template=feature-request.md&title=).

## How does the Census differentiate from other services?

The Census differentiates from existing single-cell services by providing access to the largest corpus of standardized single-cell data from CZ CELLxGENE Discover via [TileDB-SOMA](https://github.com/single-cell-data/TileDB-SOMA/issues/new/choose). 

Thus, single-cell data from about 50 mi cells across >60 K genes, with 11 standardized cell metadata variables and harmonized GENCODE annotations is at your finger tips to:

- Open and read data at low latency from the cloud.
- Query and access data using metadata filters.
- Load and create AnnData objects.
- Load and create Seurat objects.
- From Python create PyArrow objects, SciPy sparse matrices, NumPy arrays, and pandas data frames.
- From R create R Arrow objects, sparse matrices (via the Matrix package), and standard data frames and (dense) matrices.


## Can I query human and mouse data in a single query?

At the moment it is not possible to query both mouse and human data in one query. This is due to the data from these organisms using different [organism-specific gene annotations](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#required-gene-annotations).

## Where are the Census data hosted?

The Census data is publicly hosted free-of-cost in an Amazon Web Services (AWS) S3 bucket in the [`us-west-2` region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions).

## How can I increase the performance of my queries?

Since the access patterns are via the internet, usually the main limiting step for data queries is bandwidth and client location.

We recommend the following to increase query efficency: 

- Utilize a computer connected to high-speed internet.
- Utilize an ethernet connection and not a wifi connection.
- If possible utilize online computing located in the west coast of the US.
   - Highly recommended: [EC2 AWS instances](https://aws.amazon.com/ec2/) in the `us-west-2` region.

## Can I use conda to install the Census Python API?

Currently there is not conda package available for `cellxgene-census`. However you can use conda in combination with `pip` to install the package in a conda environment:

```bash
conda create -n census_env python=3.10
conda activate census_env
pip install cellxgene-census
```

## How can I ask for support?

You can either submit a [github issue ](https://github.com/chanzuckerberg/cellxgene-census/issues/new/choose)or join post in the slack channel `#cellxgene-census-users` at the [CZI Slack community](https://cziscience.slack.com/join/shared_invite/zt-czl1kp2v-sgGpY4RxO3bPYmFg2XlbZA#/shared-invite/email)

## How can I ask for new features?

You can submit a [feature request in the github repository](https://github.com/chanzuckerberg/cellxgene-census/issues/new?assignees=&labels=user+request&template=feature-request.md&title=).

## How can I contribute my data to the Census?

You first need to submit your data to [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/), then it will automatically get included in the Census if it meets the [biological criteria defined in the Census schema](https://github.com/chanzuckerberg/cellxgene-census/blob/main/docs/cellxgene_census_schema.md#data-included). 

To submit your data to CZ CELLxGENE Discover you need to follow these [instructions](https://cellxgene.cziscience.com/docs/032__Contribute%20and%20Publish%20Data). 

## Why do I get an `ArraySchema` error when opening the Census?

You may get this error if you are trying to open a Census data build with an old version of the Census API. Please update your Python or R Census package.

If the error persists please file a [github issue ](https://github.com/chanzuckerberg/cellxgene-census/issues/new/choose).

