# Census supports categoricals for cell metadata

*Published:* *April 4th, 2024*

*By:* *[Emanuele Bezzi](ebezzi@chanzuckerberg.com)* & [Pablo Garcia-Nieto](pgarcia-nieto@chanzuckerberg.com)

Starting from the `2024-04-01` Census build, a subset of the columns in the `obs` dataframe are now categorical instead of strings. 

Overall users will observe a smaller memory footprint when loading Census data into memory. 🚀

However, this may break some existing pipelines as explained below.

## Potential breaking changes

For **Python users**, note that Pandas will encode these columns as `pandas.Categorical`  for which some downstream operations may need to be adapted. See [this link](https://pandas.pydata.org/docs/user_guide/categorical.html#operations) for more details. In particular:

> Series methods like Series.value_counts() will use all categories, even if some categories are not present in the data

and 

> DataFrame methods like sum, groupby, pivot, value_counts also show “unused” categories when observed=False, which is the default.

For **R users**, note that these columns will be encoded as `factor` and similarly downstream operations may need to be adapted. See [this link](https://r4ds.had.co.nz/factors.html) for more details.

For **Python and R users** interfacing with `arrow`, these columns will be encoded as `dictionary`, see more details for R in [this link](https://arrow.apache.org/docs/r/reference/dictionary.html) and Python in [this link](https://arrow.apache.org/docs/python/generated/pyarrow.dictionary.html).


## Identifying the `obs` columns encoded as categorical

Users can always check the the type of each cell metadata variable by inspecting the schema of `obs`. Categoricals will be shown as `dictionary`.

In Python:

```python
import cellxgene_census
census = cellxgene_census.open_soma(census_version="latest")
census["census_data"]["homo_sapiens"].obs.schema

# soma_joinid: int64 not null
# dataset_id: dictionary<values=string, indices=int16, ordered=0> not null
# assay: dictionary<values=string, indices=int8, ordered=0> not null
# assay_ontology_term_id: dictionary<values=string, indices=int8, ordered=0> not null
# cell_type: dictionary<values=string, indices=int16, ordered=0> not null
# cell_type_ontology_term_id: dictionary<values=string, indices=int16, ordered=0> not null
# development_stage: dictionary<values=string, indices=int16, ordered=0> not null
# development_stage_ontology_term_id: dictionary<values=string, indices=int16, 
# [OUTPUT TRUNCATED]
``` 

In R:

```r
library("cellxgene.census")
census = open_soma(census_version="latest")
census$get("census_data")$get("homo_sapiens")$obs$schema()

# Schema
# soma_joinid: int64 not null
# dataset_id: dictionary<values=string, indices=int16> not null
# assay: dictionary<values=string, indices=int8> not null
# assay_ontology_term_id: dictionary<values=string, indices=int8> not null
# cell_type: dictionary<values=string, indices=int16> not null
# cell_type_ontology_term_id: dictionary<values=string, indices=int16> not null
# development_stage: dictionary<values=string, indices=int16> not null
# development_stage_ontology_term_id: dictionary<values=string, indices=int16> not null
# [OUTPUT TRUNCATED]
```


