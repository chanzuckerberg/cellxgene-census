# Memory-efficient implementations of commonly used single-cell methods

*Published:* *September 18, 2023*

*By:* *[Pablo Garcia-Nieto](pgarcia-nieto@chanzuckerberg.com)*

The Census team is thrilled to officially announce memory-efficient implementations of some of the most widely used single-cell algorithms.

With just a few lines of code, using the Census Python API, users can now perform the following computational tasks across tens of millions of cells using a conventional laptop with 8GB of memory:

* Calculating average and variance gene expression for cells or genes. See example below or the full [tutorial here](../../notebooks/experimental/mean_variance.ipynb).
* Obtaining batch-corrected highly variable genes.  See example below or the full [tutorial here](../../notebooks/experimental/highly_variable_genes).

These implementations are interwoven with the way users query slices of Census data, which means that these tasks can be seamlessly applied to any slice of the 33M+ cells available in Census.

Continue reading for more implementation details and usage examples.

## Efficient calculation of average and variance gene expression across millions of cells

With `cellxgene_census.experimental.pp.mean_variance` users can now get gene expression average and variance for all genes or cells in a given Census query.

### How it works

Calculations are done in an accumulative incremental fashion, meaning that only a small fraction of the total data is processed at any given time.

The Census data is downloaded in increments and average and variance accumulators are updated at each incremental step. The implementation also takes advantage of CPU-based multiprocessing to speed up the process.

Currently, the mean and variance are calculated using the full population of cells/genes, including those with a zero valued measurement. In the future, we will enable calculation of mean including only the population of non-zero cells/genes.

### Example: *KRAS* and *AQP4* average and variance expression in lung epithelial cells

The following calculates the average and variance values for the genes *KRAS* and *AQP4* in all epithelial cells of the human lung.

Users can easily switch the calculation, and obtain average and variance for each cell across the genes in the query. This is controlled by the `axis` argument of `mean_variance`.

```python
import cellxgene_census
import tiledbsoma as soma
from cellxgene_census.experimental.pp import mean_variance

# Open the Census
census = cellxgene_census.open_soma()
human_data = census["census_data"]["homo_sapiens"]

# Set filters
cell_filter = (
  "is_primary_data == True "
  "and tissue_general == 'lung' "
  "and cell_type == 'epithelial cell'"
 )
gene_filter = "feature_name in ['KRAS', 'AQP4']"

# Perform query
query = human_data.axis_query(
  measurement_name="RNA",
  obs_query=soma.AxisQuery(value_filter= cell_filter),
  var_query=soma.AxisQuery(value_filter= gene_filter)
)

# Calculate mean and average per gene
mean_variance_df = mean_variance(query, axis=0, calculate_mean=True, calculate_variance=True)

# Get gene metadata of query
gene_df = query.var().concat().to_pandas()

query.close()
census.close()
```

Which results in:

```python
mean_variance_df
# soma_joinid      mean     variance
# 8624         3.071926  5741.242485
# 16437        8.233282   452.119153

gene_df
#    soma_joinid       feature_id feature_name  feature_length
# 0         8624  ENSG00000171885         AQP4            5943
# 1        16437  ENSG00000133703         KRAS            6845
```

## Efficient calculation of highly variable genes across millions of cells

With `cellxgene_census.experimental.pp.get_highly_variable_genes` users can get the most highly variable genes of a Census query while accounting for batch effects.

This is usually the first pre-processing step necessary for other downstream tasks, for example data integration.

### How it works

The Census algorithm is based on the scanpy method `scanpy.pp.highly_variable_genes`, and in particular the Seurat V3 method, which is designed for raw counts and can account for batch effects.

The Census implementation utilizes the same incremental paradigm used in `cellxgene_census.experimental.pp.mean_variance` (see above), calculating incremental-based mean and variance accumulators with some tweaks to comply to the Seurat V3 method.

### Example: Finding highly variable genes for all cells of the human esophagus

The following example identifies the top 1000 highly variable genes for all human esophagus cells. As a general rule of thumb it is good to use `dataset_id` as the batch variable.

```python
import cellxgene_census
from cellxgene_census.experimental.pp import get_highly_variable_genes

census = cellxgene_census.open_soma()

hvg = get_highly_variable_genes(
  census,
  organism="Homo sapiens",
  obs_value_filter="is_primary_data == True and tissue_general == 'esophagus'",
  n_top_genes = 1000,
  batch_key = "dataset_id"
)

census.close()
```

Which results in:

```python
hvg
# soma_joinid    means  variances  ...  variances_norm  highly_variable
# 0            0.003692   0.004627  ...        0.748221            False
# 1            0.003084   0.003203  ...        0.898657            False
# 2            0.014962   0.037395  ...        0.513473            False
# 3            0.218865   1.547648  ...        4.786928             True
# 4            0.002142   0.002242  ...        0.894955            False
# ...               ...        ...  ...             ...              ...
# 60659        0.000000   0.000000  ...        0.000000            False
# 60660        0.000000   0.000000  ...        0.000000            False
# 60661        0.000000   0.000000  ...        0.000000            False
# 60662        0.000000   0.000000  ...        0.000000            False
# 60663        0.000000   0.000000  ...        0.000000            False
```
