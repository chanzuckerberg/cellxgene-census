# CZ CELLxGENE Discover Census in AWS

The single-cell data from [CZ CELLxGENE Discover Census](cellxgene_census_docsite_landing.md) are available for public access via Amazon Web Services (AWS).

This page describes what Census data are available in AWS and how to access them.

Contents

- [Census data available in AWS](#census-data-available-in-aws)
- [How to access AWS Census data](#how-to-access-aws-census-data)

## Census data available in AWS

The single-cell data from CZ CELLxGENE Discover included in Census (see [inclusion criteria](cellxgene_census_docsite_schema.md#data-included-in-the-census)) are available either as Census-wide TileDB files or individual H5AD files of the source datasets.

### Data specifications

<table class="custom-table">
<thead>
    <th>Data</th>
    <th>Format</th>
    <th>Access API</th>
    <th>Data schema</th>
    <th>Root S3 bucket</th>
    <th>Regions</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Census-wide</td>
    <td rowspan="2"><a href="https://github.com/TileDB-Inc/TileDB">TileDB</a></td>
    <td><a href="https://github.com/chanzuckerberg/cellxgene-census/tree/main">CELLxGENE-Census</a></td>
    <td rowspan="2"><a href="https://github.com/chanzuckerberg/cellxgene-census/blob/main/docs/cellxgene_census_schema.md">CZ CELLxGENE Discover <b>Census</b> Schema</a></td>
    <td rowspan="2">s3://cellxgene-census-public-us-west-2/cell-census/<code>[tag]</code>/soma/</td>
    <td rowspan="3">us-west-2</td>
  </tr>
  <tr>
    <td><a href="https://github.com/single-cell-data/TileDB-SOMA">TileDB-SOMA</a></td>
  </tr>
  <tr>
    <td rowspan>Source datasets</td>
    <td rowspan"><a href="https://anndata.readthedocs.io/en/latest/fileformat-prose.html#elements">H5AD</a></td>
    <td><a href="https://anndata.readthedocs.io/en/latest/index.html">AnnData</a></td>
    <td rowspan><a href="https://github.com/chanzuckerberg/single-cell-curation/tree/main/schema">CZ CELLxGENE Discover <b>Dataset</b> Schema</a></td>
    <td rowspan>s3://cellxgene-census-public-us-west-2/cell-census/<code>[tag]</code>/h5ads/</td>
  </tr>
</tbody>
</table>

See the next section for a definition of `[tag]`.

### Data release versioning

A data release is a Census build that is publicly hosted in AWS. A Census build is a TileDB-SOMA collection and its corresponding source H5AD files with the Census data from CZ CELLxGENE Discover.

Any given Census build is named with a unique `[tag]`, normally the date of build, e.g. "2023-05-15".

The are two types of data releases:

- Long-Term Supported (LTS).
- Weekly.

For more information and for a list of all LTS Census data releases available please refer to [Census data releases](cellxgene_census_docsite_data_release_info.md).

## How to access AWS Census data

### AWS CLI for programatic downloads

Users can bulk-download Census data via the [AWS CLI](https://aws.amazon.com/cli/).

For example, to download the H5ADs files of the Census LTS release `2023-07-25`, users can execute the following from a shell session:

```bash
aws s3 sync --no-sign-request s3://cellxgene-census-public-us-west-2/cell-census/2023-07-25/h5ads/ ./h5ads/
```

And to download the TileDB files:

```bash
aws s3 sync --no-sign-request s3:/cellxgene-census-public-us-west-2/cell-census/2023-07-25/soma/ ./soma/
```

### CELLxGENE Census API (Python and R)

This is the recommend method for accessing Census data. Please follow the [Census API quick start guide](cellxgene_census_docsite_quick_start.md) for a full guide.

For example, in Python users can create an iterator for the cell metadata Data Frame as follows:

``` python
import cellxgene_census

with cellxgene_census.open_soma() as census:
    cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
        value_filter = "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
        column_names = ["assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease"]
    )
```

If a **local copy** of the Census data exists, users can access it by providing the path to the `soma/` folder.

``` python
import cellxgene_census

with cellxgene_census.open_soma(uri="local/path/to/soma/") as census:
   ...
```

If a copy of the Census data exists in a **private S3 bucket**, users can access it by providing the URI `soma/`
folder in the S3 bucket. This will also require customizing TileDB configuration options to specify the
bucket's AWS region and that signed requests should be used for S3 API calls. This can be done as follows:

``` python
import cellxgene_census

uri = "s3://my-private-data-bucket/cell-census/2023-07-25/soma/"

tiledb_config={"vfs.s3.no_sign_request": "false",
               "vfs.s3.region": "us-east-1"}

with cellxgene_census.open_soma(uri=uri, tiledb_config=tiledb_config) as census:
   ...
```

### TileDB-SOMA API (Python and R)

The Census API provides convenience wrappers for TileDB-SOMA to access the Census Data hosted at AWS. Users can interact directly with the Census TileDB data directly via the TileDB-SOMA APIs. Please refer to the [TileDb-SOMA documentation](https://tiledbsoma.readthedocs.io/en/latest/) for full details on usage.

For example, in Python users can create an iterator for the cell metadata Data Frame as follows:

``` python
import cellxgene_census
import tiledbsoma

uri = "s3://cellxgene-census-public-us-west-2/cell-census/2023-07-25/soma/"
ctx = cellxgene_census.get_default_soma_context()

with tiledbsoma.open(uri, context=ctx) as census:
    cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
        value_filter = "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
        column_names = ["assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease"]
    )
```
