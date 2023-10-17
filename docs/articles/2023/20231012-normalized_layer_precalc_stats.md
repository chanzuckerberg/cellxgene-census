# Introducing a normalized layer and pre-calculated cell and gene statistics in Census

<!-- markdownlint-disable MD036 -->
*Published: October 12, 2023*

*By: [Maximilian Lombardo](mlombardo@chanzuckerberg.com) and [Pablo Garcia-Nieto](pgarcia-nieto@chanzuckerberg.com)*
<!-- markdownlint-enable MD036 -->

The Census team is happy to announce the introduction of two new data features, tailored to empower your single-cell research: a library-size, normalized expression layer and pre-calculated cell and gene statistics. This work is reflected in changes introduced in the [Census schema V1.1.0](https://github.com/chanzuckerberg/cellxgene-census/blob/main/docs/cellxgene_census_schema.md).

With these additions users can easily:

- Expand their Census query filters to select genes or cells based on the new metadata. For example, selecting only cells with N number of genes expressed.
- Export the expanded cell and gene metadata for downstream analysis.
- Export a normalized expression data matrix for downstream analysis.

These features are currently exclusive to the "latest" versions of the Census data release and they will be available in the next LTS data release. We invite your feedback as you explore these novel functionalities.

Keep on reading to find out more about these features!

## Description of new data added to Census

All of the following changes were introduced in the Census schema V1.1.0.

### Added a new library-size normalized layer

We have introduced a library-size normalized X layer for the RNA measurements of both the human and mouse experiments available as `X["normalized"]`.  The normalized layer is built by dividing each value in the raw count matrix by its corresponding row sum (i.e. size normalization).

To reduce data size and improve performance, normalized values are stored with a reduced floating point precision. In addition, to ensure that small count values do not round to zero, a small sigma has been added. You will see the effect of these artifacts in row (per-cell) values not summing to precisely 1.0.

### Enhanced gene metadata

The `ms["RNA"].var` DataFrame for both the human and mouse experiments has been enriched with two new metadata fields:

- `nnz` — the number of explicitly stored values, effectively the number of cells expressing this gene.
- `n_measured_obs` — the "measured" cells for this gene, effectively the number of cells for which this gene was measured in their respective dataset.

### Enhanced cell metadata

The `obs` DataFrame for both the human and mouse experiments is now augmented with the following new metadata, allowing users to forego common calculations used in early data pre-processing. For each cell:

- `raw_sum` — the sum of the raw counts, derived from `X["raw"]`.
- `nnz` — the number of explicitly stored values, effectively the number of genes expressed on this cell.
- `raw_mean_nnz` — the average counts from explicitly stored values.
- `raw_variance_nnz` — the variance of the counts from explicitly stored values.
- `n_measured_vars` — the "measured" genes, effectively the number of genes measured in the dataset from which the cell originated, thus all cells from the same dataset have the same value for this variable.

## How to use the new features

### Exporting the normalized data to existing single-cell toolkits

In Python, the normalized data can be exported into AnnData specifying the `X_name = "normalized"` argument of the `cellxgene.get_anndata()` method.

```python
import cellxgene_census

with cellxgene_census.open_soma(census_version = "latest") as census
    adata = cellxgene_census.get_anndata
        census = census,
        organism = "Homo sapiens",
        var_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']",
        obs_value_filter = "cell_type == 'sympathetic neuron'",
        column_names = {"obs": ["tissue", "cell_type"]},
        X_name = "normalized" # Specify the normalized layer for this query
    )
```

Similarly, in R we can export the data to Seurat or SingleCellExperiment objects with the argument `X_layers` of the functions `get_seurat()` and `get_single_cell_experiment()`.

```r
library("cellxgene.census")
library("Seurat")

census <- open_soma(census_version = "latest")

organism <- "Homo sapiens"
gene_filter <- "feature_id %in% c('ENSG00000107317', 'ENSG00000106034')"
cell_filter <-  "cell_type == 'sympathetic neuron'"
cell_columns <- c("tissue", "cell_type")
layers <- c(data = "normalized")

seurat_obj <- get_seurat(
   census = census,
   organism = organism,
   var_value_filter = gene_filter,
   obs_value_filter = cell_filter,
   obs_column_names = cell_columns,
   X_layers = layers
)

#Single Cell Experiment

library("SingleCellExperiment")

sce_obj <- get_single_cell_experiment(
   census = census,
   organism = organism,
   var_value_filter = gene_filter,
   obs_value_filter = cell_filter,
   obs_column_names = cell_columns,
   X_layers = layers
)

```

### Accessing library-size normalized data layer via TileDB-SOMA

For memory-efficient data retrieval, you can use TileDB-SOMA as outlined below. In Python this looks like the following.

```python

import cellxgene_census
import tiledbsoma

# Open context manager
with cellxgene_census.open_soma(census_version = "latest") as census:

    # Access human SOMA object
    human = census["census_data"]["homo_sapiens"]

    query = human.axis_query(
       measurement_name = "RNA",
       obs_query = tiledbsoma.AxisQuery(
           value_filter = "tissue == 'brain' and sex == 'male'"
       )
    )

    # Set iterable for normalized matrix
    iterator = query.X("normalized").tables()
    
    # Iterate over the normalized matrix.
    # Get an iterative slice as pyarrow.Table
    raw_slice = next(iterator)
    
    # Perform analysis
    
    # close the query
    query.close()
```

And the equivalent code in R.

```r
library("cellxgene.census")
library("tiledbsoma")

human <-  census$get("census_data")$get("homo_sapiens")
query <-  human$axis_query(
  measurement_name = "RNA",
  obs_query = SOMAAxisQuery$new(
    value_filter = "tissue == 'brain' & sex == 'male'"
  )
)

# Set iterable for normalized matrix
iterator <-  query$X("normalized")$tables()

# Iterate over the normalized matrix.
# Get an iterative slice as an Arrow Table
raw_slice <-  iterator$read_next()

# Perform analysis
```

### Utilizing pre-calculated stats for querying `obs` and `var`

To filter cells or genes based on pre-calculated statistics and export to AnnData, you can use the new metadata variables as value filters.

For example, you can add a filter to query cells with more than 500 genes expressed, along with other filters. In Python this looks like the following.

```python
import cellxgene_census

with cellxgene_census.open_soma(census_version = "latest") as census:
    adata = cellxgene_census.get_anndata(
        census = census,
        organism = "Homo sapiens",
        obs_value_filter = "nnz > 500 and cell_type == 'sympathetic neuron'",
        column_names = {"obs": ["tissue", "cell_type"]},
        var_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']",
    )

    print(adata)
```

In R, the equivalent code looks as follows.

```r
#Seurat
library("cellxgene.census")
library("Seurat")

census <- open_soma(census_version = "latest")

organism <- "Homo sapiens"
gene_filter <- "feature_id %in% c('ENSG00000107317', 'ENSG00000106034')"
cell_filter <-  "nnz > 500 & cell_type == 'sympathetic neuron'"
cell_columns <- c("tissue", "cell_type")

seurat_obj <- get_seurat(
   census = census,
   organism = organism,
   var_value_filter = gene_filter,
   obs_value_filter = cell_filter,
   obs_column_names = cell_columns
)

#Single Cell Experiment

library("SingleCellExperiment")

sce_obj <- get_single_cell_experiment(
   census = census,
   organism = organism,
   var_value_filter = gene_filter,
   obs_value_filter = cell_filter,
   obs_column_names = cell_columns
)
```

## Help us improve these data additions

We encourage you to engage with these new features in the Census API and share your feedback. This input is invaluable for the ongoing enhancement of the Census project.

For further information on any new feature, please reach out to us at [soma@chanzuckerberg.com](soma@chanzuckerberg.com). To report issues or for additional feedback, refer to our [Census GitHub repository](https://github.com/chanzuckerberg/cellxgene-census/issues).
