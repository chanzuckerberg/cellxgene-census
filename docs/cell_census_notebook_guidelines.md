# Cell Census API notebook/vignette guidelines

API demonstration code that is part of the documentation should be deposited here: 

- Python notebooks [`cell-census/api/python/notebooks`](https://github.com/chanzuckerberg/cell-census/tree/main/api/python/notebooks)
- R vignettes [`cell-census/api/r/CellCensus/vignettes`](https://github.com/chanzuckerberg/cell-census/tree/main/api/r/CellCensus/vignettes)


Since these assets are user-facing and are one of the main form users get onboarded with the product, then the following guidelines need to be followed to ensure readability and a consistent experience.

## Guidelines

### Title

* It must use the highest-level markdown header `#`.
* Unless needed it, it should not contain the feature name in it.
* It should be concise, self-explanatory and if possible indicate an action.

Examples:

:white_check_mark: `# Exploring all data from a tissue.`

:white_check_mark: `# Query and extract slices of data.`

:x: `# Census datasets presence` *not self-explanatory, has "Census" in it.*

:x: `# Normalizing full-length gene sequencing data from the Cell Censuse` *it has "Census" in it.*



## Examples 