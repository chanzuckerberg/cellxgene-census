# Cell Census API notebook/vignette editorial guidelines

API demonstration code that is part of the documentation should be deposited here: 

- Python notebooks [`cellxgene-census/api/python/notebooks`](https://github.com/chanzuckerberg/cellxgene-census/tree/main/api/python/notebooks)
- R vignettes [`cellxgene-census/api/r/CellCensus/vignettes`](https://github.com/chanzuckerberg/cellxgene-census/tree/main/api/r/cellxgene.census/vignettes)

To include Python notebooks in the doc site, create a symlink to the notebook in [cellxgene-census/docs/notebooks/](https://github.com/chanzuckerberg/cellxgene-census/tree/main/docs/notebooks) folder.

These assets are user-facing and are automatically rendered to the doc-sites and are one of the primary means by which users get onboarded to the product. Thus the following guidelines need to be followed to ensure readability and a consistent experience.

## Guidelines

### Title

* It must use the highest-level markdown header `#`.
* Unless needed, it should not contain any direct mentions of "Census".
* It should be concise, self-explanatory, and if possible indicate an action.

Examples:

:white_check_mark: `# Exploring all data from a tissue.`

:white_check_mark: `# Query and extract slices of data.`

:x: `# Census datasets presence` *not self-explanatory, has "Census" in it.*

:x: `# Normalizing full-length gene sequencing data from the Cell Census` *has "Census" in it.*

### Introduction

Introductory text must be included right underneath the title.

* It must provide a one paragraph summary of the notebook's goals.
* It must not contain an explanation of the Census.

Examples: 

:white_check_mark:

> This notebook provides a demonstration for integrating two Census datasets using scvi-tools. The goal is not to provide an exhaustive guide on proper integration, but to showcase what information in the Census can inform data integration.

:x: *it contains a long explanation of what the Census is, and the goal is not clear*

> The Census is a versioned container for the single-cell data hosted at CELLxGENE Discover. The Census utilizes SOMA powered by TileDB for storing, accessing, and efficiently filtering data.
>
>This notebook shows you how to learn about the Census contents and how to query it.

### Table of Contents 

Immediately after the introduction a table of contents must be provided:

* It must be placed under the bolded word "**Contents**" . 
* It must contain an ordered list of the second-level headers (`##`) used for [Sections](#sections).
* If necessary it may contain sub-lists corresponding to lower-level headers (`###`, etc)

Example:

:white_check_mark:

> **Contents**
> 
> 1. Learning about the lung data.
> 2. Fetching all human lung data from the Census.
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
Census datasets using `scvi-tools`. The goal is not to 
provide an exhaustive guide on proper integration, but to showcase 
what information in the Census can inform data integration.

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
  import cellxgene_census
  import scanpy as sc
  import numpy as np
  import scvi
  from scipy.sparse import csr_matrix
\code 

Now we can open the Census 

\code 
  census = cellxgene_census.open_soma(census_version="latest")
\code

In this notebook we will use Tabula Muris Senis data 
from the liver as it contains cells from both 10X 
Genomics and Smart-Seq2 technologies.

Let's query the datasets table of the Census by 
filtering on collection_name for "Tabula Muris Senis" 
and dataset_title for "liver".

[...]
```
