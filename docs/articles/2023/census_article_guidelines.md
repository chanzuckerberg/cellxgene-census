# R package `cellxgene.census` v1.0.0 is out!

*Published: TBD August 2023*

*By: [Pablo Garcia-Nieto](pgarcia-nieto@chanzuckerberg.com)*

The Census team is pleased to announce the release of the R package `cellxgene.census`. 🎉 🎉 

This has been long coming since our Python release back in May. Now, from R computational biologists can access the Census data which is the largest harmonized aggregation of single-cell data, composed of >30M cells and >60K genes.
 
With `cellxgene.census` in a few seconds users can slice and access Census data using cell or gene filters across hundreds of datasets. The data can be fetched in an iterative fashion for bigger-than-memory slices of data, or quickly exported to basic R structures, and [Seurat](https://satijalab.org/seurat/) or [SingleCellExperiment](https://bioconductor.org/packages/release/bioc/html/SingleCellExperiment.html) for downstream analysis.

## Installation and usage

Users can install `cellxgene.census` and its dependencies following the [installation instructions](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_installation.html).

To learn more about the package please make sure to check out the following resources:

* [Quick start guide.](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_quick_start.html)
* [R reference docs and tutorials.](https://chanzuckerberg.github.io/cellxgene-census/r/index.html)
* Querying and slicing data tutorial << INSERT LINK >>.

## Census R package is made possible by `tiledbsoma`

The `cellxgene.census` package relies on [TileDB-SOMA](https://github.com/single-cell-data/TileDB-SOMA) R's package `tiledbsoma` for all of its data access capabilities as shown in the next section. 

CZI and TileDB have worked closely on the development of `tiledbsoma` and recently promoted it from beta stage to its first stable version.  << INSERT link their release notes >>

## Efficient access to single-cell data for >30M cells from R

Census hosts ever-increasing [data releases](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_data_release_info.html) from CZ CELLxGENE Discover, representing the largest aggregation of standardized single-cell data. 

Census data are accompanied by cell and gene metadata that have been harmonized on ontologies across all datasets hosted in CZ CELLxGENE Discover. For example all cell types and tissues have been mapped to a value of the CL and UBERON ontologies, respectively. You can find more about the data in the [Census data and schema](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_schema.html) page.

With the R package `cellxgene.census` researchers can have access to all of these data and metadata directly from an R session with following capabilities:

### Easy-to-use handles to the cloud-hosted Census data

From R users can get handle to the data by opening the census.

```r
library("cellxgene.census")

census <- open_soma()

# Your work!

census.close()
``` 

### Querying and reading single-cell metadata from Census

Following our [Census data and schema](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_schema.html), users can navigate and query Census data and metadata by using any combination of gene and cell filters.

For example, reading a slice of the human cell metadata for about 300K cells with Microglial cells or Neurons from female donors :

```r
library("cellxgene.census")

census <- open_soma()

# Open obs SOMADataFrame
cell_metadata = census$get("census_data")$get("homo_sapiens")$get("obs")

# Read as an iterator of Arrow Tables
cell_metadata = cell_metadata$read(
   value_filter = "sex == 'female' & cell_type %in% c('microglial cell', 'neuron')",
   column_names = c("assay", "cell_type", "sex", "tissue", "tissue_general", "suspension_type", "disease")
)

# Concatenate iterations
cell_metadata <- cell_metadata$concat()

# Convert to R tibble (dataframe)
cell_metadata = as.data.frame(cell_metadata)

census.close()
```

### Exporting Census slices to `Seurat` and `SingleCellExperiment`

Similarly querying both the single-cell data along metadata can be easily exported to  `Seurat` or `SingleCellExperiment` object for downstream analysis:

```r
library("cellxgene.census")

census = open_soma()

organism = "Homo sapiens"
gene_filter = "feature_id %in% c('ENSG00000161798', 'ENSG00000188229')"
cell_filter =  "sex == 'female' & cell_type %in% c('microglial cell', 'neuron')"
cell_columns = c("assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease")

seurat_obj = get_seurat(census = census, organism = organism, var_value_filter = gene_filter, obs_value_filter = cell_filter, obs_column_names = cell_columns)

sce_obj = get_single_cell_experiment(census = census, organism = organism, var_value_filter = gene_filter, obs_value_filter = cell_filter, obs_column_names = cell_columns)

census.close()
```

### Streaming data incrementally in chunks

Sometimes Census queries can be too large to be loaded in memory. TileDB-SOMA allows users to query Census data in an incremental fashion using iterators.

To find out more about iterable-based queries you can check out the following resources:

* [Memory-efficient queries from R.](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_quick_start.html#id2)
* [The SOMA objects overview from TileDB-SOMA.](https://single-cell-data.github.io/TileDB-SOMA/articles/soma-objects.html)
