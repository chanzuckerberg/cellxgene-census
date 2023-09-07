
# R package of CZ CELLxGENE Discover Census

<!-- badges: start -->
<!-- badges: end -->

This is the documentation for the R package `cellxgene.census` which is part of CZ CELLxGENE Discover Census. For full details on Census data and capabilities please go to the [main Census site](https://chanzuckerberg.github.io/cellxgene-census/).

`cellxgene.census` provides an API to efficiently access the cloud-hosted Census single-cell data from R. In just a few seconds users can access any slice of Census data using cell or gene filters across hundreds of single-cell datasets.

Census data can be fetched in an iterative fashion for bigger-than-memory slices of data, or quickly exported to basic R structures, as well as `Seurat` or `SingleCellExperiment` objects for downstream analysis.

## Installation

If installing from **Ubuntu**, you may need to install the following libraries via `apt install`,  `libxml2-dev` `libssl-dev` `libcurl4-openssl-dev`. In addition you must have `cmake` v3.21 or greater.

If installing from **MacOS**, you will need to install the [developer tools `Xcode`](https://apps.apple.com/us/app/xcode/id497799835?mt=12).

**Windows** is not supported.

Then in an R session install `cellxgene.census` from R-Universe.

```r
install.packages(
  "cellxgene.census",
  repos=c('https://chanzuckerberg.r-universe.dev', 'https://cloud.r-project.org')
)
```

To be able to export Census data to `Seurat` or `SingleCellExperiment` you also need to install their respective packages.

```r
# Seurat
install.packages("Seurat")

# SingleCellExperiment
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("SingleCellExperiment")
```

## Usage

Check out the vignettes in the "Articles" section of the navigation bar on this site. We highly recommend the following vignettes as a starting point:

* [Querying and fetching the single-cell data and cell/gene metadata](./articles/census_query_extract.html)
* [Learning about the CZ CELLxGENE Discover Census](./articles/comp_bio_census_info.html)

You can also check out out the [quick start guide](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_quick_start.html) in the main Census site.

### Example `Seurat` and `SingleCellExperiment` query

The following creates a `Seurat` object on-demand with all sympathetic neurons in Census and filtering only for the genes `ENSG00000161798`, `ENSG00000188229`.

```r
library("cellxgene.census")
library("Seurat")

census <- open_soma()

organism <- "Homo sapiens"
gene_filter <- "feature_id %in% c('ENSG00000107317', 'ENSG00000106034')"
cell_filter <-  "cell_type == 'sympathetic neuron'"
cell_columns <- c("assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease")

seurat_obj <- get_seurat(
   census = census,
   organism = organism,
   var_value_filter = gene_filter,
   obs_value_filter = cell_filter,
   obs_column_names = cell_columns
)
```

And the following retrieves the same data as a `SingleCellExperiment` object.

```r
library("SingleCellExperiment")

sce_obj <- get_single_cell_experiment(
   census = census,
   organism = organism,
   var_value_filter = gene_filter,
   obs_value_filter = cell_filter,
   obs_column_names = cell_columns
)
```

## For More Help

For more help, please go visit the [main Census site](https://chanzuckerberg.github.io/cellxgene-census/).

If you believe you have found a security issue, we would appreciate notification. Please send an email to <security@chanzuckerberg.com>.
