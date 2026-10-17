# CZ CELLxGENE Discover Census

The `cellxgene_census` package provides an API to facilitate the use of the CZ CELLxGENE Discover Census. For more information about the API and the project visit the [chanzuckerberg/cellxgene-census GitHub repo](https://github.com/chanzuckerberg/cellxgene-census/).

## For More Help

For more help, please file a issue on the repo, or contact us at <soma@chanzuckerberg.com>.

If you believe you have found a security issue, we would appreciate notification. Please send email to <security@chanzuckerberg.com>.

## Development Environment Setup

- Create a virtual environment using `venv` or `conda`
- `cd` to the root of this repository
- `pip install -e api/python/cellxgene_census`
- To install dependencies needed to work on the [experimental](./src/cellxgene_census/experimental/) portion of the API:
  `pip install -e 'api/python/cellxgene_census[experimental]'`.
- `pip install jupyterlab`
- **Test it!** Either open up a new `jupyter` notebook or the `python` interpreter and run this code:

```python
import cellxgene_census

with cellxgene_census.open_soma() as census:

    cell_metadata = cellxgene_census.get_obs(
        census,
        "homo_sapiens",
        value_filter = "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
        column_names = ["assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease"]
    )
    cell_metadata
```

The output is a `pandas.DataFrame` with over 600K cells meeting our query criteria and the selected columns:

```python

The "stable" release is currently 2023-12-15. Specify 'census_version="2023-12-15"' in future calls to open_soma() to ensure data consistency.

                assay        cell_type                 tissue tissue_general suspension_type disease     sex
0        Smart-seq v4  microglial cell  middle temporal gyrus          brain         nucleus  normal  female
1        Smart-seq v4  microglial cell  middle temporal gyrus          brain         nucleus  normal  female
2        Smart-seq v4  microglial cell  middle temporal gyrus          brain         nucleus  normal  female
3        Smart-seq v4  microglial cell  middle temporal gyrus          brain         nucleus  normal  female
4        Smart-seq v4  microglial cell  middle temporal gyrus          brain         nucleus  normal  female
...               ...              ...                    ...            ...             ...     ...     ...
607636  microwell-seq           neuron          adrenal gland  adrenal gland            cell  normal  female
607637  microwell-seq           neuron          adrenal gland  adrenal gland            cell  normal  female
607638  microwell-seq           neuron          adrenal gland  adrenal gland            cell  normal  female
607639  microwell-seq           neuron          adrenal gland  adrenal gland            cell  normal  female
607640  microwell-seq           neuron          adrenal gland  adrenal gland            cell  normal  female

[607641 rows x 7 columns]

```

- Learn more about the Census API by going through the tutorials in the [notebooks](../notebooks/)

## Restricting embedding search to a biological reference population

`experimental.find_nearest_obs` accepts `obs_value_filter` in the same SOMA
predicate syntax as `get_anndata`. For example, query only primary lung cells:

```python
neighbors = cellxgene_census.experimental.find_nearest_obs(
    "scvi",
    "homo_sapiens",
    "2025-11-08",
    query_adata,  # matching scVI vectors in query_adata.obsm["scvi"]
    obs_value_filter="tissue_general == 'lung' and is_primary_data == True",
    k=10,
    memory_GiB=4,
)
```

Providing a filter selects exact search over the complete, finite stored
embeddings of cells satisfying that predicate in the selected release and
organism. Filtering is applied before ranking, so an eligible neighbor can be
found even when arbitrarily many excluded cells are closer. Results use squared
Euclidean distances, the same units as the default Census IVFFlat indexes, with
float64 query conversion and distance accumulation, and float32 output. Ranking
uses float64-computed distances; equal internal distances are ordered by ascending
`soma_joinid`. Distinct internal distances can round to the same reported float32
value, so ties in the returned distance values need not be join-ID ordered.

This mode reads the selected embedding vectors, not the expression matrix or a
model, and does not require a precomputed nearest-neighbor index. It scans every
eligible embedding and costs O(number of queries × selected cells × embedding
dimensions) distance work; broad filters may be slow. Metadata IDs are streamed,
embedding reads and distance computations are chunked, and only running top-k
results are retained. `memory_GiB` budgets numerical working arrays and results,
with 64 MiB of additional I/O headroom; it is not a hard process-RSS limit.
Caller-owned query arrays, streamed Arrow tables, and TileDB internal allocations
are outside this numerical budget. Initial TileDB read buffers are set to 8 MiB
each, but returned Arrow batches may be larger and include predicate columns.
Only join IDs are retained from each batch during processing. SOMA's
`batch_size` is currently ignored upstream, so the implementation does not rely
on it as a memory bound.

Missing, partial, NaN, or infinite reference embeddings are excluded. Fewer than
`k` eligible finite embeddings raises `ValueError` rather than returning invalid
neighbor IDs. Query vectors must be dense numeric NumPy arrays representable as
finite float64 values, `k` and
`memory_GiB` must be positive integers, and a filter must be nonempty. Squared
distances too large for finite float32 output also raise `ValueError`. Predicate
syntax errors are reported by SOMA. `nprobe` is irrelevant for exact search and
additional index query keyword arguments are rejected.
Empty query batches return arrays of shape `(0, k)` without scanning candidates,
even when fewer than `k` reference cells exist; predicate syntax and stored asset
metadata are not read in this no-op case.

Omitting `obs_value_filter` retains the existing approximate IVFFlat search and
its behavior. Exactness here refers to the stored embeddings, which may already
have reduced precision; it does not recover an unrounded model representation.
