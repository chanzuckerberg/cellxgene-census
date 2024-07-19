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
