# Introducing a normalized layer and pre-calculated cell and gene statistics in Census.

*Published: 25 August 2023*  
*By: [Author Name](mailto:author1@chanzuckerberg.com), [Co-Author](mailto:author2@chanzuckerberg.com)*

The Census team is thrilled to announce the introduction of Census schema V1.1.0. This version offers enhanced metadata and API functionalities, tailored to empower your single-cell research.

These features are currently experimental and exclusive to the "latest" version of Census. We invite your feedback as you explore these novel functionalities.

## Schema Changes in V1.1.0

### Added `dataset_version_id` to Census Table

The new field `dataset_version_id` has been introduced in `census_obj["census_info"]["datasets"]` to facilitate dataset versioning and management.

### Added New Library-Size Normalized Layer in X["normalized"]

We've introduced a library-size normalized layer in X["normalized"]. A small sigma value has been consistently added to each position, ensuring that no explicit zeros are recorded.

### Enhanced Metadata Fields in `ms["RNA"].var` DataFrame

The `ms["RNA"].var` DataFrame has been enriched with two new metadata fields: `nnz` and `n_measured_obs`, which provide a count of non-zero values and "measured" cells, respectively.

### Enhanced Metadata Fields in `ms["RNA"].obs` DataFrame

The `obs` DataFrame is now augmented with the following new metadata, allowing users to forego common calculations used in early data pre-processing:

- `raw_sum`: Represents the count sum derived from X["raw"]
- `nnz`: Enumerates the number of non-zero (nnz) values
- `raw_mean`: Provides the average count of nnz values
- `raw_variance`: Indicates the count variance of nnz values
- `n_measured_vars`: Enumerates the "measured" genes, determined by the sum of the presence matrix

## How to Use the New Features

### Exporting Normalized Data

Normalized data can be exported into AnnData with the following code:

```python
import cellxgene_census

with cellxgene_census.open_soma() as census:
    adata = cellxgene_census.get_anndata(
        census = census,
        organism = "Homo sapiens",
        var_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']",
        obs_value_filter = "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
        column_names = {"obs": ["assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease"]},
        X_name = "normalized" # Specificy the normalized layer for this query
        
    )
    
    print(adata)

#Adata with normalized layer
```

### Accessing Library-Size Normalized Data Layer via TileDB-SOMA 

For memory-efficient data retrieval, you can use TileDB-SOMA as outlined below:

```python
# Memory-efficient data retrieval

import cellxgene_census
import tiledbsoma

# Open context manager
with cellxgene_census.open_soma() as census:

    # Access human SOMA object
    human = census["census_data"]["homo_sapiens"]

    query = human.axis_query(
       measurement_name = "RNA",
       obs_query = tiledbsoma.AxisQuery(
           value_filter = "tissue == 'brain' and sex == 'male'"
       )...
    )

    # Set iterable
    iterator = query.X("normalized").tables()
    
    # Iterate over the matrix count. Get an iterative slice as pyarrow.Table
    raw_slice = next (iterator)

    # close the query
    query.close()


```

### Utilizing Pre-Calculated Stats for Querying `obs` and `var`

To filter based on pre-calculated statistics and export to AnnData, execute the following:

```python
# Example code to fetch all cells with more than 5 detected genes as an AnnData object

import cellxgene_census

with cellxgene_census.open_soma() as census:
    adata = cellxgene_census.get_anndata(
        census = census,
        organism = "Homo sapiens",
        var_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']",
        obs_value_filter = "nnz > '5' and sex == 'female' and cell_type in ['microglial cell', 'neuron']", 
        column_names = {"obs": ["assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease"]},
    )

    print(adata)

#adata with filtered cells
```

---

We encourage you to engage with these new features in the Census API and share your feedback. This input is invaluable for the ongoing enhancement of the Census project.

For further information on the new library-size normalized layer, please reach out to us at [cellxgene@chanzuckerberg.com](cellxgene@chanzuckerberg.com). To report issues or for additional feedback, refer to our [Census GitHub repository](https://github.com/chanzuckerberg/cellxgene-census/issues).

---
