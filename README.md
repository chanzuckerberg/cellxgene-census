[![codecov](https://codecov.io/gh/chanzuckerberg/cell-census/branch/main/graph/badge.svg?token=byX1pyDlc9)](https://codecov.io/gh/chanzuckerberg/cell-census)

# Cell Census of CZ CELLxGENE Discover

**CZ CELLxGENE Discover** ([cellxgene.cziscience.com](https://cellxgene.cziscience.com/)) is a free-to-use data portal hosting a growing corpus of more than **700 single-cell datasets** comprising about **50 million cells** from the major human and mouse tissues. The portal provides a set of visual tools to download and explore the data. **All data is standardized** to include raw counts and a common vocabulary for gene and cell metadata.

The **Cell Census** provides easy-to-use and efficient computational tooling to access, query, and analyze all RNA data from CZ CELLxGENE Discover. The Cell Census aims to break the barrier of data fragmentation in the single-cell field by presenting a **new access paradigm of cell-based slicing and querying** for all data at CZ CELLxGENE Discover.

## Motivation: Single-cell analysis at scale 

The **Cell Census** is a data object publicly hosted online and a convenience API to open it. The object is built using the [SOMA](https://github.com/single-cell-data/SOMA) API and data model via its implementation [TileDB-SOMA](https://github.com/single-cell-data/TileDB-SOMA). As such, the Cell Census has all the data capabilities offered by TileDB-SOMA including:

- Cloud-based data storage and access.
- Efficient access for larger-than-memory slices of data.
- Data streaming for iterative/parallelizablne  methods.
- R and Python support.
- Export to AnnData and Seurat.


The Cell Census is free to use.

## Cell Census data releases

Starting in  mid 2023, Cell Census long-term supported data builds will be released every 6 months and are guaranteed to be stored for public access for at least 5 years upon release. 

In between long-term supported data build releases, weekly builds are released without any guarantee of permanence. 

## Cell Census data organization

The Cell Census follows a specific [data schema](https://github.com/chanzuckerberg/cell-census/blob/main/docs/cell_census_schema_0.1.0.md). Briefly, the Cell Census is a collection of a variety of **SOMA** objects organized with the following hierarchy.


Cell Census, a collection with:

- `"census_info" ` — collection with summary objects:
   - `"summary"` — data frame with Cell Census metadata.
   - `"datasets"` — data frame listing all datasets included.
   - `"summary_cell_counts"`  — data frame with cell counts across cell metadata variables.
- `"census_data"` — collection with the single-cell data per organism:
	- `"homo_sapiens"` or `"mus_musculus"` — collection with:
		- `obs`  — data frame with cell metadata.
		- `ms["RNA"]` — collection with the count matrix and gene metadata:
		   - `X["raw"]` — sparse matrix with raw counts.
		   - `var` — data frame with gene metadata for >60K genes.
		   - `"feature_dataset_presence_matrix"`— sparse boolean matrix flagging genes measured per dataset. 

## Getting started

### Requirements

The Cell Census requires a Linux or MacOS system with:

- Python 3.7 to Python 3.10. Or R, supported versions TBD.
- Recommended: >16 GB of memory.
- Recommended: >5 Mbps internet connection. 

### Documentation

Reference documentation, data description, and tutorials can be access at the Cell Census doc-site. *Coming soon*. 

Demonstration notebooks can be found [here](https://github.com/chanzuckerberg/cell-census/tree/main/api/python/notebooks).

### Python quick start

#### Installation

It is recommended to install the Cell Census and all of its dependencies in a new virtual environment via `pip`:

```
$ python -m venv ./venv
$ source ./venv/bin/activate
$ pip install -U cell-census
```

#### Usage examples

##### Opening the Cell Census

You can directly open the Cell Census.

```python
import cell_census
census = cell_census.open_soma()
...
census.close()
```

Or use a context manager.

```python
import cell_census
with cell_census.open_soma() as census:
   ...
```

##### Querying a slice of cell metadata.

The following filters female cells of cell type "microglial cell" or "neuron", and selects the columns "assay", "cell_type" and "tissue".

```python
# Reads SOMA data frame
cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
   value_filter = "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
   column_names = ["assay", "cell_type", "tissue"]
)

# Concatenates results to pyarrow.Table
cell_metadata = cell_metadata.concat()

# Converts to pandas.DataFrame
cell_metadata = cell_metadata.to_pandas()
```

##### Obtaining a slice as AnnData 

The following filters female cells of cell type "microglial cell" or "neuron", and selects the cell metadata columns "assay", "cell_type" and "tissue". It also filters for the genes "ENSG00000161798" and "ENSG00000188229".

```python
adata = cell_census.get_anndata(
    census = census,
    organism = "Homo sapiens",
    var_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']",
    obs_value_filter = "cell_type == 'B cell' and tissue_general == 'lung' and disease == 'COVID-19'",
    column_names = {"obs": ["assay", "cell_type", "tissue"]},
)

```

##### Memory-efficient queries

This example provides a demonstration to accessed the data for larger-than-memory operations using **TileDB-SOMA** operations. 

First we initiate a lazy-evaluation query to access all brain and male cells from human. This query needs to be closed — `query.close()` — or used called in a context manager — `with ...`.

```
import tiledbsoma

human = census["census_data"]["homo_sapiens"]

query = human.axis_query(
   measurement_name = "RNA",
   obs_query = tiledbsoma.AxisQuery(
      value_filter = "tissue == 'brain' and sex == 'male'"
   )
)
```

Now we can iterate over the matrix count, as well as the cell and gene metadata. For example, to iterate over the matrix count, we can get an iterator and perform operations for each iteration.

```
iterator = query.X("raw").tables()

# Get a slice a pyarrow.Table
raw_slice = next (iterator) 
...
```

Alternatively.

```
for raw_slice in  query.X("raw").tables():
   ...
``` 

And you must close the query and Cell Census after you are done.

```
query.close()
census.close()
```

### R quick start

*Coming soon*


## Questions, feedback and issues

- Questions: we encourage you to ask questions via [github issues](https://github.com/chanzuckerberg/cell-census/issues). Alternatively, for quick support you can join the C[ZI Science Community](https://join-cellxgene-users.herokuapp.com/) on Slack and join the `#cell-census-users` channel
- Bugs: please submit a [github issue](https://github.com/chanzuckerberg/cell-census/issues). 
- Feature requests: please submit requests using this [form](https://airtable.com/shrVV1g0d6nvBoQYu).
- Security issue: if you believe you have found a security issue, we would appreciate notification. Please send an email to <security@chanzuckerberg.com>.
- You can send any other feedback to <soma@chanzuckerberg.com>


## Coming soon

- R support!
- We are currently working on creating the tooling necessary to perform data modeling at scale with seamless of integration of the Cell Census and [PyTorch](https://pytorch.org/).
- To increase the usability of the Cell Census for research, in 2023 and 2024 we are planing to explore the following areas :
   - Organism-wide normalization
   - Organism-wide embeddings
   - Smart subsampling

## Contribute

*Coming soon.*

## Projects and tools using the Cell Census

If you are interested in listing a project here, please reach out to us at <soma@chanzuckerberg.com>

## Reuse

The contents of this Github repository are freely available for reuse under the [MIT license](https://opensource.org/licenses/MIT). Data in the Cell Census are available for re-use under the [CC-BY license](https://creativecommons.org/licenses/by/4.0/).


## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to <opensource@chanzuckerberg.com>.

