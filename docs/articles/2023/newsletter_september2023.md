# Introducing Schema V1.1.0: Key Updates in Census API and Data Layer

*Published: 25 August 2023*  
*By: [Author Name](mailto:author1@chanzuckerberg.com), [Co-Author](mailto:author2@chanzuckerberg.com)*

The Census team is thrilled to announce the introduction of Census schema V1.1.0. This version offers enhanced data and API functionalities, tailored to empower your single-cell research. 

These features are currently experimental and exclusive to the "latest" version of Census. We invite your feedback as you explore these novel functionalities.

## Schema Changes in V1.1.0

### Adds `dataset_version_id` to Census Table

The new field `dataset_version_id` has been introduced in `census_obj["census_info"]["datasets"]` to facilitate dataset versioning and management.

### Numerical Precision in X["normalized"] Layer

The X["normalized"] layer now boasts improved numerical precision. A small sigma value has been consistently added to each position, ensuring that no explicit zeros are recorded.

### New Columns in `ms["RNA"].var` DataFrame

The `ms["RNA"].var` dataframe has been enriched with two new columns: `nnz` and `n_measured_obs`, which provide a count of non-zero values and "measured" cells, respectively.

### Enhanced `obs` DataFrame 

The `obs` dataframe is now augmented with the following new columns, each serving a specific computational purpose:

- `raw_sum`: Represents the count sum derived from X["raw"]
- `nnz`: Enumerates the number of non-zero (nnz) values
- `raw_mean`: Provides the average count of nnz values
- `raw_variance`: Indicates the count variance of nnz values
- `n_measured_vars`: Enumerates the "measured" genes, determined by the sum of the presence matrix

## How to Use the New Features

### Exporting Normalized Data

Normalized data can be exported into AnnData and Seurat with the following code:

```python
# Example code
```

### Accessing Data via TileDB-SOMA 

For a memory-efficient data retrieval, you can use TileDB-SOMA as outlined below:

```python
# Example code
```

### Utilizing Pre-Calculated Stats for Querying `obs` and `var`

To filter based on pre-calculated statistics and export to AnnData, execute the following:

```python
# Example code to fetch all cell/genes with a count greater than 5
```

---

We encourage you to engage with these advancements and share your feedback. This input is invaluable for the ongoing enhancement of the Census project.

For further information on numerical precision improvements, please consult with @pablo-gar or @bkmartinjr. To report issues or for additional feedback, refer to our Census GitHub repository.

---
