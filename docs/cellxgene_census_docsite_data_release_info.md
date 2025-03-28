# Census data releases

**Last edited**: Feb 13th, 2024.

**Contents:**

1. [What is a Census data release?](#what-is-a-census-data-release)
1. [List of LTS Census data releases](#list-of-lts-census-data-releases)
1. [Compatibility with package versions](#compatibility-with-package-versions)

## What is a Census data release?

It is a Census build that is publicly hosted online. A Census build is
a [TileDB-SOMA](https://github.com/single-cell-data/TileDB-SOMA) collection with the Census data from [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/) as specified in the [Census schema](cellxgene_census_docsite_schema.md).

Any given Census build is named with a unique tag, normally the date of build, e.g., `"2025-01-30"`.

### Long-term supported (LTS) Census releases

To enable data stability and scientific reproducibility, [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/) plans to perform regular LTS Census data releases:

* Published online every six months for public access, starting on May 15, 2023.
* Available for public access for at least 5 years upon publication.

The most recent LTS Census data release is the default opened by the APIs and recognized as `census_version = "stable"`. To open previous LTS Census data releases, you can directly specify the version via its build date `census_version = "[YYYY]-[MM]-[DD]"`.

Python

```python
import cellxgene_census
census = cellxgene_census.open_soma(census_version = "stable")
```

R

```r
library("cellxgene.census")
census <- open_soma(census_version = "stable")
```

### Weekly Census releases (latest)

[CZ CELLxGENE Discover](https://cellxgene.cziscience.com/) ingests a handful of new datasets every week. To quickly enable access to these new data via the Census, CZ CELLxGENE Discover plans to perform weekly Census data releases:

* Available for public access for 1 month.

The most recent weekly release can be opened by the APIs by specifying `census_version = "latest"`.

Python

```python
import cellxgene_census
census = cellxgene_census.open_soma(census_version = "latest")
```

R

```r
library("cellxgene.census")
census <- open_soma(census_version = "latest")
```

## List of LTS Census data releases

### LTS 2025-01-30

Open this data release by specifying `census_version = "2025-01-30"` in future calls to `open_soma()`.

#### Version information

| Information                       | Value                                                                                 |
|-----------------------------------|---------------------------------------------------------------------------------------|
| Census schema version             | [**2.1.0**](https://github.com/chanzuckerberg/cellxgene-census/blob/main/docs/cellxgene_census_schema.md) |
| Census build date                 | **2025-01-30**                                                                        |
| Dataset schema version            | [**5.2.0**](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/5.2.0/schema.md)      |
| Number of datasets                | **1573**                                                                               |

#### Cell and donor counts

| Type              | _Homo sapiens_ | _Mus musculus_ |
|-------------------|----------------|----------------|
| Total cells       | 109,085,698    | 45,351,496     |
| Unique cells      | 65,601,657     | 20,208,302     |

#### Cell metadata

| Category                | _Homo sapiens_ | _Mus musculus_ |
|:------------------------|---------------:|---------------:|
| Assay                   |             31 |             17 |
| Cell type               |            827 |            453 |
| Development stage       |            179 |             58 |
| Disease                 |            140 |             12 |
| Self-reported ethnicity |             36 |              1 |
| Sex                     |              3 |              3 |
| Suspension type         |              1 |              1 |
| Tissue                  |            379 |             99 |
| Tissue general          |             68 |             36 |

#### Embeddings

Find out more in the [Census model page](https://cellxgene.cziscience.com/census-models).

Available embeddings can be accessed via [`cellxgene_census.experimental.get_embedding()`](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.experimental.get_embedding.html#cellxgene_census.experimental.get_embedding), or by specifying the `obs_embeddings`/`var_embeddings` field in [`cellxgene_census.get_anndata()`](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.get_anndata.html#cellxgene_census.get_anndata).

##### Cells

| Method                    | _Homo sapiens_ | _Mus musculus_ |
|---------------------------|----------------|----------------|
| scVI                      | `scvi`         | `scvi`         |

### LTS 2024-07-01

Open this data release by specifying `census_version = "2024-07-01"` in future calls to `open_soma()`.

#### Version information

| Information                       | Value      |
|-----------------------------------|------------|
| Census schema version             | [2.0.1](https://github.com/chanzuckerberg/cellxgene-census/blob/fad674674e5070b735a29bc069d1d3dc21d2e5e8/docs/cellxgene_census_schema.md) |
| Census build date                 | 2024-05-20 |
| Dataset schema version            | [5.0.0](https://github.com/chanzuckerberg/cellxgene-census/blob/fad674674e5070b735a29bc069d1d3dc21d2e5e8/docs/cellxgene_census_schema.md)      |
| Number of datasets                | 812        |

#### Cell and donor counts

| Type              | _Homo sapiens_ | _Mus musculus_ |
|-------------------|----------------|----------------|
| Total cells       | 74,322,510     | 41,233,630     |
| Unique cells      | 44,265,932     | 16,332,034     |

#### Cell metadata

| Category                | _Homo sapiens_ | _Mus musculus_ |
|-------------------------|----------------|----------------|
| Assay                   | 24             | 11             |
| Cell type               | 698            | 364            |
| Development stage       | 176            | 48             |
| Disease                 | 109            | 7              |
| Self-reported ethnicity | 31             | _NA_           |
| Sex                     | 3              | 3              |
| Suspension type         | 2              | 2              |
| Tissue                  | 267            | 84             |
| Tissue general          | 55             | 29             |

#### Embeddings

Find out more in the [Census model page](https://cellxgene.cziscience.com/census-models).

Available embeddings can be accessed via [`cellxgene_census.experimental.get_embedding()`](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.experimental.get_embedding.html#cellxgene_census.experimental.get_embedding), or by specifying the `obs_embeddings`/`var_embeddings` field in [`cellxgene_census.get_anndata()`](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.get_anndata.html#cellxgene_census.get_anndata).

##### Cells

| Method                    | _Homo sapiens_ | _Mus musculus_ |
|---------------------------|----------------|----------------|
| scVI                      | `scvi`         | `scvi`         |
| Geneformer                | `geneformer`   | _NA_           |

### LTS 2023-12-15

Open this data release by specifying `census_version = "2023-12-15"` in future calls to `open_soma()`.

#### Version information

| Information                       | Value      |
|-----------------------------------|------------|
| Census schema version             | [1.2.0](https://github.com/chanzuckerberg/cellxgene-census/blob/3ff1033135b3a9365c239a9442798d88aae94d03/docs/cellxgene_census_schema.md) |
| Census build date                 | 2023-12-15 |
| Dataset schema version            | [3.1.0](https://github.com/chanzuckerberg/single-cell-curation/blob/8ae36ef3fb5a826511dc657d1b8c6d4a772d32e8/schema/3.1.0/schema.md)      |
| Number of datasets                | 651        |

#### Cell and donor counts

| Type              | _Homo sapiens_ | _Mus musculus_ |
|-------------------|----------------|----------------|
| Total cells       | 62,998,417     | 5,684,805      |
| Unique cells      | 36,227,903     | 4,128,230     |

#### Cell metadata

| Category                | _Homo sapiens_ | _Mus musculus_ |
|-------------------------|----------------|----------------|
| Assay                   | 20             | 10              |
| Cell type               | 631            | 248            |
| Development stage       | 173            | 36             |
| Disease                 | 72             | 5              |
| Self-reported ethnicity | 30             | _NA_           |
| Sex                     | 3              | 3              |
| Suspension type         | 2              | 2              |
| Tissue                  | 230            | 74             |
| Tissue general          | 53             | 27             |

#### Embeddings

Find out more in the [Census model page](https://cellxgene.cziscience.com/census-models).

Available embeddings can be accessed via [`cellxgene_census.experimental.get_embedding()`](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.experimental.get_embedding.html#cellxgene_census.experimental.get_embedding), or by specifying the `obs_embeddings`/`var_embeddings` field in [`cellxgene_census.get_anndata()`](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.get_anndata.html#cellxgene_census.get_anndata).

##### Cells

| Method                    | _Homo sapiens_ | _Mus musculus_ |
|---------------------------|----------------|----------------|
| scVI                      | `scvi`         | `scvi`         |
| Fine-tuned Geneformer     | `geneformer`   | _NA_           |
| scGPT                     | `scgpt`        | _NA_           |
| Universal Cell Embeddings | `uce`          | _NA_           |
| NMF                       | `nmf`          | _NA_           |

##### Features

| Method                    | _Homo sapiens_ | _Mus musculus_ |
|---------------------------|----------------|----------------|
| NMF                       | `nmf`          | _NA_           |

### LTS 2023-07-25

Open this data release by specifying `census_version = "2023-07-25"` in future calls to `open_soma()`.

#### Version information

| Information                       | Value      |
|-----------------------------------|------------|
| Census schema version             | [1.0.0](https://github.com/chanzuckerberg/cellxgene-census/blob/f06bcebb6471735681fd84734d2d581c44e049e7/docs/cellxgene_census_schema.md) |
| Census build date                 | 2023-07-25 |
| Dataset schema version            | [3.0.0](https://github.com/chanzuckerberg/single-cell-curation/blob/a64ac9eb70e3e777ee34098ae82120c2d21692b0/schema/3.0.0/schema.md)      |
| Number of datasets                | 593        |

#### Cell and donor counts

| Type              | _Homo sapiens_ | _Mus musculus_ |
|-------------------|----------------|----------------|
| Total cells       | 56,400,873     | 5,255,245      |
| Unique cells      | 33,364,242     | 4,083,531     |

#### Cell metadata

| Category                | _Homo sapiens_ | _Mus musculus_ |
|-------------------------|----------------|----------------|
| Assay                   | 19             | 9              |
| Cell type               | 613            | 248            |
| Development stage       | 164            | 33             |
| Disease                 | 64             | 5              |
| Self-reported ethnicity | 26             | _NA_           |
| Sex                     | 3              | 3              |
| Suspension type         | 2              | 2              |
| Tissue                  | 220            | 66             |
| Tissue general          | 54             | 27             |

### LTS 2023-05-15

Open this data release by specifying `census_version = "2023-05-15"` in future calls to `open_soma()`.

#### 🔴 Errata 🔴

##### Duplicate observations with  `is_primary_data = True`

In order to prevent duplicate data in analyses, each observation (cell) should be marked `is_primary data = True` exactly once in the Census. Since this LTS release, 243,569 observations have been identified that are represented at least twice with `is_primary_data = True`.

This issue will be corrected in the following LTS data release, by identifying and marking only one cell out of the duplicates as  `is_primary_data = True`.

If you wish to use this data release, you can consider filtering out all of these 243,569 cells by using the `soma_joinids` provided in this file [duplicate_cells_census_LTS_2023-05-15.csv.zip](https://github.com/chanzuckerberg/cellxgene-census/raw/773edab79bbdc78eccb26ec4f8211a9b4c98a71a/tools/cell_dup_check/duplicate_cells_census_LTS_2023-05-15.csv.zip). You can filter specific cells by using the `value_filter` or `obs_value_filter` of the querying API functions, for more information follow this [tutorial](https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_query_extract.html).

#### Version information

| Information                       | Value      |
|-----------------------------------|------------|
| Census schema version             | [1.0.0](https://github.com/chanzuckerberg/cellxgene-census/blob/f06bcebb6471735681fd84734d2d581c44e049e7/docs/cellxgene_census_schema.md) |
| Census build date                 | 2023-05-15 |
| Dataset schema version            | [3.0.0](https://github.com/chanzuckerberg/single-cell-curation/blob/a64ac9eb70e3e777ee34098ae82120c2d21692b0/schema/3.0.0/schema.md)      |
| Number of datasets                | 562        |

#### Cell and donor counts

| Type              | _Homo sapiens_ | _Mus musculus_ |
|-------------------|----------------|----------------|
| Total cells       | 53,794,728     | 4,086,032      |
| Unique cells      | 33,758,887     | 2,914,318      |

#### Cell metadata

| Category                | _Homo sapiens_ | _Mus musculus_ |
|-------------------------|----------------|----------------|
| Assay                   | 20             | 9              |
| Cell type               | 604            | 226            |
| Development stage       | 164            | 30             |
| Disease                 | 68             | 5              |
| Self-reported ethnicity | 26             | _NA_           |
| Sex                     | 3              | 3              |
| Suspension type         | 2              | 2              |
| Tissue                  | 227            | 51             |
| Tissue general          | 61             | 27             |

## Compatibility with package versions

Due to the nature of the Census storage backend, the format version will change from time to time. Format upgrades are always backwards compatible, but they're not always forwards compatible, which means that reading a recent Census data version using an older version of the package might result in an error.
We aim to guarantee the following policy:

* Every Census package version released after an LTS will be able to read _every_ Census data release until the next LTS.

The current LTS release (2023-12-15) is compatible with the following package versions:

* 1.10.x
* 1.11.x
* 1.12.x
* 1.13.x
