# Quick start

This page provides details to start using the Census. Click [here] (examples.rst) for more detailed Python tutorials (R vignettes coming soon).

**Contents**

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

### Querying a slice of cell metadata.

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

The output is a `pandas.DataFrame` with about 300K cells meeting our query criteria and the selected columns.

```bash
            assay        cell_type           tissue tissue_general suspension_type disease     sex
0       10x 3' v3  microglial cell              eye            eye            cell  normal  female
1       10x 3' v3  microglial cell              eye            eye            cell  normal  female
2       10x 3' v3  microglial cell              eye            eye            cell  normal  female
3       10x 3' v3  microglial cell              eye            eye            cell  normal  female
4       10x 3' v3  microglial cell              eye            eye            cell  normal  female
...           ...              ...              ...            ...             ...     ...     ...
299617  10x 3' v3           neuron  cerebral cortex          brain         nucleus  normal  female
299618  10x 3' v3           neuron  cerebral cortex          brain         nucleus  normal  female
299619  10x 3' v3           neuron  cerebral cortex          brain         nucleus  normal  female
299620  10x 3' v3           neuron  cerebral cortex          brain         nucleus  normal  female
299621  10x 3' v3           neuron  cerebral cortex          brain         nucleus  normal  female

[299622 rows x 7 columns]
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
AnnData object with n_obs × n_vars = 299622 × 2
    obs: 'assay', 'cell_type', 'tissue', 'tissue_general', 'suspension_type', 'disease', 'sex'
    var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length'
```

### Memory-efficient queries

This example provides a demonstration to access the data for larger-than-memory operations using **TileDB-SOMA** operations. 

First we initiate a lazy-evaluation query to access all brain and male cells from human. This query needs to be closed — `query.close()` — or called in a context manager — `with ...`.

```python
import cellxgene_census

with cellxgene_census.open_soma() as census:
    
    human = census["census_data"]["homo_sapiens"]
    query = human.axis_query(
    measurement_name = "RNA",
    obs_query = tiledbsoma.AxisQuery(
        value_filter = "tissue == 'brain' and sex == 'male'"
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

```
    # Continued from above
    query.close()
```

## R quick start

❗ **API is in beta and under rapid development.**

Below are 3 examples of common operations you can do with the Census. As a reminder, the reference documentation for the API can be accessed via `?`:

```r
library("cellxgene.census")

?cellxgene.census::get_seurat
```

### Querying a slice of cell metadata.

The following reads the cell metadata and filters `female` cells of cell type `microglial cell` or `neuron`, and selects the columns `assay`, `cell_type`, `tissue`, `tissue_general`, `suspension_type`, and `disease`.

The `cellxgene.census` package uses [R6](https://r6.r-lib.org/articles/Introduction.html) classes and we recommend you to get familiar with their usage. 

```r
library("cellxgene.census")

census = open_soma()

# Open obs SOMADataFrame
cell_metadata = census$get("census_data")$get("homo_sapiens")$get("obs")

# Read as Arrow Table
cell_metadata = cell_metadata$read(
   value_filter = "sex == 'female' & cell_type %in% c('microglial cell', 'neuron')",
   column_names = c("assay", "cell_type", "sex", "tissue", "tissue_general", "suspension_type", "disease")
)

# Convert to R tibble (dataframe)
cell_metadata = as.data.frame(cell_metadata)

print(cell_metadata)
```

The output is a `tibble` with about 300K cells meeting our query criteria and the selected columns.

```bash
# A tibble: 305,735 × 7
   assay     cell_type sex    tissue tissue_general suspension_type disease
   <chr>     <chr>     <chr>  <chr>  <chr>          <chr>           <chr>  
 1 10x 3' v3 neuron    female lung   lung           nucleus         normal 
 2 10x 3' v3 neuron    female lung   lung           nucleus         normal 
 3 10x 3' v3 neuron    female lung   lung           nucleus         normal 
 4 10x 3' v3 neuron    female lung   lung           nucleus         normal 
 5 10x 3' v3 neuron    female lung   lung           nucleus         normal 
 6 10x 3' v3 neuron    female lung   lung           nucleus         normal 
 7 10x 3' v3 neuron    female lung   lung           nucleus         normal 
 8 10x 3' v3 neuron    female lung   lung           nucleus         normal 
 9 10x 3' v3 neuron    female lung   lung           nucleus         normal 
10 10x 3' v3 neuron    female lung   lung           nucleus         normal 
# ℹ 305,725 more rows
# ℹ Use `print(n = ...)` to see more rows
```

### Obtaining a slice as a Seurat object 

The following creates an Seurat object on-demand with the smaller set of cells  and filtering only the genes `ENSG00000161798`, `ENSG00000188229`.

```python
library("cellxgene.census")

census = open_soma()

seurat_obj = get_seurat(
   census = census,
   organism = "Homo sapiens",
   var_value_filter = "feature_id %in% c('ENSG00000161798', 'ENSG00000188229')",
   obs_value_filter = "sex == 'female' & cell_type %in% c('microglial cell', 'neuron')",
   obs_column_names = c("assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease")
)

print(seurat_obj)
```

The output with about 5K cells and 2 genes can be now used for downstream analysis using [Seurat](https://satijalab.org/seurat/).

``` shell
An object of class Seurat 
2 features across 5876 samples within 1 assay 
Active assay: RNA (2 features, 0 variable features)
```

### Memory-efficient queries

Memory-efficient capabilities of the R API are still under active development. 

