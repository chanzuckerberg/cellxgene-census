# Quick start

This page provides details to start using the Census. Click [here](examples.rst) for more detailed Python tutorials (R vignettes coming soon).

**Contents:**

1. [Installation](#installation).
2. [Python quick start](python-quick-start).
3. [R quick start](r-quick-start).

## Installation

Install the Census API by following [these instructions.](cellxgene_census_docsite_installation.md)

## Python quick start

Below are 3 examples of common operations you can do with the Census. As a reminder, the reference documentation for the API can be accessed via `help()`:

```python
import cellxgene_census

help(cellxgene_census)
help(cellxgene_census.get_anndata)
# etc
```

### Querying a slice of cell metadata

The following reads the cell metadata and filters `female` cells of cell type `microglial cell` or `neuron`, and selects the columns `assay`, `cell_type`, `tissue`, `tissue_general`, `suspension_type`, and `disease`.

```python
import cellxgene_census

with cellxgene_census.open_soma() as census:

    # Reads SOMADataFrame as a slice
    cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
        value_filter = "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
        column_names = ["assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease"]
    )

    # Concatenates results to pyarrow.Table
    cell_metadata = cell_metadata.concat()

    # Converts to pandas.DataFrame
    cell_metadata = cell_metadata.to_pandas()

    print(cell_metadata)
```

The output is a `pandas.DataFrame` with over 300K cells meeting our query criteria and the selected columns.

```bash
The "stable" release is currently 2023-07-25. Specify 'census_version="2023-07-25"' in future calls to open_soma() to ensure data consistency.
                assay        cell_type         tissue tissue_general suspension_type disease     sex
0           10x 3' v3  microglial cell            eye            eye            cell  normal  female
1           10x 3' v3  microglial cell            eye            eye            cell  normal  female
2           10x 3' v3  microglial cell            eye            eye            cell  normal  female
3           10x 3' v3  microglial cell            eye            eye            cell  normal  female
4           10x 3' v3  microglial cell            eye            eye            cell  normal  female
...               ...              ...            ...            ...             ...     ...     ...
379219  microwell-seq           neuron  adrenal gland  adrenal gland            cell  normal  female
379220  microwell-seq           neuron  adrenal gland  adrenal gland            cell  normal  female
379221  microwell-seq           neuron  adrenal gland  adrenal gland            cell  normal  female
379222  microwell-seq           neuron  adrenal gland  adrenal gland            cell  normal  female
379223  microwell-seq           neuron  adrenal gland  adrenal gland            cell  normal  female

[379224 rows x 7 columns]
```

### Obtaining a slice as AnnData

The following creates an `anndata.AnnData` object on-demand with the same cell filtering criteria as above and filtering only the genes `ENSG00000161798`, `ENSG00000188229`.

```python
import cellxgene_census

with cellxgene_census.open_soma() as census:
    adata = cellxgene_census.get_anndata(
        census = census,
        organism = "Homo sapiens",
        var_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']",
        obs_value_filter = "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
        column_names = {"obs": ["assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease"]},
    )

    print(adata)
```

The output with about 300K cells and 2 genes can be now used for downstream analysis using [scanpy](https://scanpy.readthedocs.io/en/stable/).

``` bash
AnnData object with n_obs × n_vars = 379224 × 2
    obs: 'assay', 'cell_type', 'tissue', 'tissue_general', 'suspension_type', 'disease', 'sex'
    var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length'
```

### Memory-efficient queries

This example provides a demonstration to access the data for larger-than-memory operations using **TileDB-SOMA** operations.

First we initiate a lazy-evaluation query to access all brain and male cells from human. This query needs to be closed — `query.close()` — or called in a context manager — `with ...`.

```python
import cellxgene_census
import tiledbsoma

with cellxgene_census.open_soma() as census:

    human = census["census_data"]["homo_sapiens"]
    query = human.axis_query(
       measurement_name = "RNA",
       obs_query = tiledbsoma.AxisQuery(
           value_filter = "tissue == 'brain' and sex == 'male'"
       )
    )

    # Continued below

```

Now we can iterate over the matrix count, as well as the cell and gene metadata. For example, to iterate over the matrix count, we can get an iterator and perform operations for each iteration.

```python
    # Continued from above

    iterator = query.X("raw").tables()

    # Get an iterative slice as pyarrow.Table
    raw_slice = next (iterator)
    ...
```

And you can now perform operations on each iteration slice. As with any any Python iterator this logic can be wrapped around a `for` loop.

And you must close the query.

```python
    # Continued from above
    query.close()
```

## R quick start

Below are 3 examples of common operations you can do with the Census. As a reminder, the reference documentation for the API can be accessed via `?`:

```r
library("cellxgene.census")

?cellxgene.census::get_seurat
```

### Querying a slice of cell metadata

The following reads the cell metadata and filters `female` cells of cell type `microglial cell` or `neuron`, and selects the columns `assay`, `cell_type`, `tissue`, `tissue_general`, `suspension_type`, and `disease`.

The `cellxgene.census` package uses [R6](https://r6.r-lib.org/articles/Introduction.html) classes and we recommend you to get familiar with their usage.

```r
library("cellxgene.census")

census <- open_soma()

# Open obs SOMADataFrame
cell_metadata <-  census$get("census_data")$get("homo_sapiens")$get("obs")

# Read as Arrow Table
cell_metadata <-  cell_metadata$read(
   value_filter = "sex == 'female' & cell_type %in% c('microglial cell', 'neuron')",
   column_names = c("assay", "cell_type", "sex", "tissue", "tissue_general", "suspension_type", "disease")
)

# Concatenates results to an Arrow Table
cell_metadata <-  cell_metadata$concat()

# Convert to R tibble (dataframe)
cell_metadata <-  as.data.frame(cell_metadata)

print(cell_metadata)

census$close()
```

The output is a `tibble` with over 300K cells meeting our query criteria and the selected columns.

```bash
# A tibble: 379,224 × 7
   assay     cell_type       sex   tissue tissue_general suspension_type disease
   <chr>     <chr>           <chr> <chr>  <chr>          <chr>           <chr>
 1 10x 3' v3 microglial cell fema… eye    eye            cell            normal
 2 10x 3' v3 microglial cell fema… eye    eye            cell            normal
 3 10x 3' v3 microglial cell fema… eye    eye            cell            normal
 4 10x 3' v3 microglial cell fema… eye    eye            cell            normal
 5 10x 3' v3 microglial cell fema… eye    eye            cell            normal
 6 10x 3' v3 microglial cell fema… eye    eye            cell            normal
 7 10x 3' v3 microglial cell fema… eye    eye            cell            normal
 8 10x 3' v3 microglial cell fema… eye    eye            cell            normal
 9 10x 3' v3 microglial cell fema… eye    eye            cell            normal
10 10x 3' v3 microglial cell fema… eye    eye            cell            normal
# ℹ 379,214 more rows
# ℹ Use `print(n = ...)` to see more rows
```

### Obtaining a slice as a `Seurat` or `SingleCellExperiment` object

The following creates a Seurat object on-demand with a smaller set of cells and filtering only the genes `ENSG00000161798`, `ENSG00000188229`.

```r
library("cellxgene.census")
library("Seurat")

census <-  open_soma()

organism <-  "Homo sapiens"
gene_filter <-  "feature_id %in% c('ENSG00000107317', 'ENSG00000106034')"
cell_filter <-   "cell_type == 'sympathetic neuron'"
cell_columns <-  c("assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease")

seurat_obj <-  get_seurat(
   census = census,
   organism = organism,
   var_value_filter = gene_filter,
   obs_value_filter = cell_filter,
   obs_column_names = cell_columns
)

print(seurat_obj)
```

The output with over 4K cells and 2 genes can be now used for downstream analysis using [Seurat](https://satijalab.org/seurat/).

```shell
An object of class Seurat
2 features across 4744 samples within 1 assay
Active assay: RNA (2 features, 0 variable features)
```

Similarly a `SingleCellExperiment` object can be created.

```r
library("SingleCellExperiment")

sce_obj <-  get_single_cell_experiment(
   census = census,
   organism = organism,
   var_value_filter = gene_filter,
   obs_value_filter = cell_filter,
   obs_column_names = cell_columns
)

print(sce_obj)
```

The output with over 4K cells and 2 genes can be now used for downstream analysis using the [Bioconductor ecosystem](https://bioconductor.org/packages/release/bioc/html/SingleCellExperiment.html).

```shell
class: SingleCellExperiment
dim: 2 4744
metadata(0):
assays(1): counts
rownames(2): ENSG00000106034 ENSG00000107317
rowData names(2): feature_name feature_length
colnames(4744): obs48350835 obs48351829 ... obs52469564 obs52470190
colData names(6): assay cell_type ... suspension_type disease
reducedDimNames(0):
mainExpName: RNA
altExpNames(0):
```

### Memory-efficient queries

This example provides a demonstration to access the data for larger-than-memory operations using **TileDB-SOMA** operations.

First we initiate a lazy-evaluation query to access all brain and male cells from human. This query needs to be closed — `query$close()`.

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

# Continued below

```

Now we can iterate over the matrix count, as well as the cell and gene metadata. For example, to iterate over the matrix count, we can get an iterator and perform operations for each iteration.

```r
# Continued from above

iterator <-  query$X("raw")$tables()
# For sparse matrices use query$X("raw")$sparse_matrix()

# Get an iterative slice as an Arrow Table
raw_slice <-  iterator$read_next()

#...
```

And you can now perform operations on each iteration slice. This logic can be wrapped around a `while()` loop and checking the iteration state by monitoring the logical output of `iterator$read_complete()`

And you must close the query and census.

```r
# Continued from above
query.close()
census.close()
```
