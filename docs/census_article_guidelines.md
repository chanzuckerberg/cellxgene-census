# Census "what's new?" article editorial guidelines

"What's new?" articles are blog-like short pieces intended to deliver information to the user on the following:

* Short descriptions of API updates.
* Short descriptions of Census data updates.
* Small description of an in-house analysis with the Census.
* Blurbs on external uses of the Census.

The goals of these articles are to have:

* Master reference articles to link for other channels (e.g. slack, twitter).
* One-stop place for users to have a historical view of Census developments and analysis.

A great example of this approach is the [Apache Arrow Blog](https://arrow.apache.org/blog/).

## Location

The articles should be placed in the folder `./docs/articles/` and further organized in year subdirectories `[YYYY]/`. The articles should be in markdown format `*.md`

The articles should be named with a date prefix followed by a name `[YYYYMMDD]-[name].md`, the name is left to the discretion of the user.

For example:

`./docs/articles/2023/20230810-r_api_is_out.md`

## Guidelines

### Title

* It must use the highest-level markdown header `#`.
* It should be concise and self-explanatory.

Examples:

:white_check_mark: `# R package cellxgene.census 1.0.0 is out`

:white_check_mark: `# A new normalized layer has been added to the Census data`

:x: `# Cool new feature is out` *not self-explanatory*

:x: `# Errors in Census data 2023-05-15` *not self-explanatory*

### Date & author

Immediately below the title, and date and author(s) should be added to the article in italics. The date should be in format "[DD] [Month] [YYYY]" and followed by the keyword "Published: ", and the author(s) must be written after the keyboard "By: " have an email link. Both must be in italics.

Example:

```markdown
*Published: 10 August 2023*

*By: [John Smith](author1@chanzuckerberg.com), [Phil Scoot](author2@chanzuckerberg.com)*
```

### Introduction

Introductory text of 1-2 paragraphs must be included right underneath the date and authors.

* It must provide a one paragraph summary of the article.
* It must not contain an explanation of the Census.

Example:

> The Census team is pleased to announce the release of the R package `cellxgene.census`, this has been long coming since our Python release back in May. Now, from R users can access the Census data which is the largest harmonized aggregation of single-cell data, composed of >30M cells and >60K genes.
>
> With `cellxgene.census` users can access Census access and slice the data using cell or gene filters across hundreds of datasets. Users can fetch the data in an iterative fashion for bigger-than-memory slices of data, or export to Seurat or SingleCellExperiment objects

### Sections

The rest of the article content must be organized within sections:

* The section title must use the second-level markdown header `##`. **This is important as the python doc-site renders these in the sidebar and in the full view of all notebooks.**
* The section title should be concise, self-explanatory.
* The section's contents and presence or absence of sub-headers are left to the discretion of the writer.

## Example article

```markdown
# R package cellxgene.census 1.0.0 is out

*Published: 10 August 2023*

*By: [Pablo Garcia-Nieto](pgarcia-nieto@chanzuckerberg.com)*

The Census team is pleased to announce the release of the R package
`cellxgene.census`, this has been long coming since our Python
release back in May. Now, from R users can access the Census data
which is the largest harmonized aggregation of single-cell data,
composed of >30M cells and >60K genes.

With `cellxgene.census` users can access Census access and slice
the data using cell or gene filters across hundreds of datasets.
Users can fetch the data in an iterative fashion for bigger-than-memory
slices of data, or export to Seurat or SingleCellExperiment objects.

## Installation & usage

Link to installation usage

## Capabitlies

List of R API capabilities

[...]
```
