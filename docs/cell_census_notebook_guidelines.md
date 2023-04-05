# Cell Census API notebook/vignette editorial guidelines

API demonstration code that is part of the documentation should be deposited here: 

- Python notebooks [`cell-census/api/python/notebooks`](https://github.com/chanzuckerberg/cell-census/tree/main/api/python/notebooks)
- R vignettes [`cell-census/api/r/CellCensus/vignettes`](https://github.com/chanzuckerberg/cell-census/tree/main/api/r/CellCensus/vignettes)


These assets are user-facing and are automatically rendered to the doc-sites, they are one of the primary means by which users get onboarded to the product. Thus the following guidelines need to be followed to ensure readability and a consistent experience.

## Guidelines

### Title

* It must use the highest-level markdown header `#`.
* Unless needed, it should not contain "Cell Census".
* It should be concise, self-explanatory, and if possible indicate an action.

Examples:

:white_check_mark: `# Exploring all data from a tissue.`

:white_check_mark: `# Query and extract slices of data.`

:x: `# Census datasets presence` *not self-explanatory, has "Census" in it.*

:x: `# Normalizing full-length gene sequencing data from the Cell Census` *has "Census" in it.*

### Introduction

Introductory text must be included right underneath the title.

* It must provide a one paragraph summary of the notebook's goals.
* It must not contain an explanation of the Cell Census.

Examples: 

:white_check_mark:

> This notebook provides a demonstration for integrating two Cell Census datasets using scvi-tools. The goal is not to provide an exhaustive guide on proper integration, but to showcase what information in the Cell Census can inform data integration.

:x: *it contains a long explanation of what the Cell Census is, and the goal is not clear*

> The Cell Census is a versioned container for the single-cell data hosted at CELLxGENE Discover. The Cell Census utilizes SOMA powered by TileDB for storing, accessing, and efficiently filtering data.
>
>This notebook shows you how to learn about the Cell Census contents and how to query it.

### Table of Contents 

Immediately after the introduction a table of contents must be provided:

* It must be placed under the bolded word "**Contents**" . 
* It must contain an ordered list of the second-level headers (`##`) used for [Sections](#Sections).
* If necessary it may contain sub-lists corresponding to lower-level headers (`###`, etc)

Example:

:white_check_mark:

> **Contents**
> 
> 1. Learning about the lung data.
> 2. Fetching all human lung data from the Cell Census.
> 3. Obtaining QC metrics for this data slice.

### Sections

The rest of the notebook/vignette content must be organized within sections:

* The section title must use the second-level markdown header `##`. **This is important as the python doc-site renders these in the sidebar and in the full view of all notebooks.**
* The section title should be concise, self-explanatory, and if possible indicate an action.
* The section's contents and presence or absence of sub-headers are left to the discretion of the writer.
* The section's non-code content should be kept as succinct as possible.


## Example notebook/vignette 

```
# Integrating data with SCVI.

This notebook provides a demonstration for integrating two 
Cell Census datasets using `scvi-tools`. The goal is not to 
provide an exhaustive guide on proper integration, but to showcase 
what information in the Cell Census can inform data integration.

**Contents**

1. Finding and fetching data from mouse liver.
2. Gene-length normalization of Smart-Seq2 data.
3. Integration with scvi-tools.
   1. Inspecting data prior to integration.
   2. Integration with batch defined as dataset_id.
   3. Integration with batch defined as dataset_id + donor_id.
   4. Integration with batch defined as dataset_id + donor_id + assay_ontology_term_id + suspension_type.

## Finding and fetching data from mouse liver

Let's load all modules needed for this notebook.

\code
  import cell_census
  import scanpy as sc
  import numpy as np
  import scvi
  from scipy.sparse import csr_matrix
\code 

Now we can open the Cell Census 

\code 
  census = cell_census.open_soma(census_version="latest")
\code

In this notebook we will use Tabula Muris Senis data 
from the liver as it contains cells from both 10X 
Genomics and Smart-Seq2 technologies.

Let's query the datasets table of the Cell Census by 
filtering on collection_name for "Tabula Muris Senis" 
and dataset_title for "liver".

[...]
```
