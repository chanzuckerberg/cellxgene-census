:exclamation: **R API in beta and unstable.**


# CZ CELLxGENE Discover Census

<p align="center">
	<img align="center" src="./cellxgene_census_docsite_workflow.svg">
</p>

The CZ CELLxGENE Discover **Census** provides efficient computational tooling to access, query, and analyze all single-cell RNA data from CZ CELLxGENE Discover. 

Using a **new access paradigm of cell-based slicing and querying**, you can interact with the data across datasets through [TileDB-SOMA](https://github.com/single-cell-data/TileDB-SOMA), or get slices in [AnnData](https://anndata.readthedocs.io/) or [Seurat](https://satijalab.org/seurat/) objects.

Get started on using the Census:

- [Installation](https://cellxgene-census.readthedocs.io/en/latest/installation.html)
- [R & Python quick start](https://cellxgene-census.readthedocs.io/en/latest/quick-start.html)
- [Census data and schema](https://cellxgene-census.readthedocs.io/en/latest/schema.html)
- [FAQ](https://cellxgene-census.readthedocs.io/en/latest/faq.html)
- [Python tutorials](https://cellxgene-census.readthedocs.io/en/latest/examples.html)
- R tutorials. *Coming soon.*


## Citing the Census

Please follow the [citation guidelines](https://cellxgene.cziscience.com/docs/08__Cite%20cellxgene%20in%20your%20publications) offered by CZ CELLxGENE Discover.

## Census Capabilities

The Census is a data object publicly hosted online and a convenience API to open it. The object is built using the [SOMA](https://github.com/single-cell-data/SOMA) API and data model via its implementation [TileDB-SOMA](https://github.com/single-cell-data/TileDB-SOMA). As such, the Census has all the data capabilities offered by TileDB-SOMA including:

**Data access at scale**

- Cloud-based data access.
- Efficient access for larger-than-memory slices of data.
- Query and access data based on cell or gene metadata at low latency.

**Interoperability with existing single-cell toolkits**

- Load and create [AnnData](https://anndata.readthedocs.io/en/latest/) objects.
- Load and create [Seurat](https://satijalab.org/seurat/) objects. Coming soon.

**Interoperability with existing Python or R data structures**

- From Python create [PyArrow](https://arrow.apache.org/docs/python/index.html) objects, SciPy sparse matrices, NumPy arrays, and pandas data frames.
- From R create [R Arrow](https://arrow.apache.org/docs/r/index.html) objects, sparse matrices (via the [Matrix](https://cran.r-project.org/package=Matrix) package), and standard data frames and (dense) matrices.

## Census Data Releases

The Census data release plans are detailed [here](https://cellxgene-census.readthedocs.io/en/latest/data_release.html). 

Shortly, starting in mid 2023 Census long-term supported data releases will be published every 6 months and will be publicly accessible for at least 5 years. In addition, weekly releases will be published without any guarantee of permanence. 


## Questions, feedback and issues

- Questions: we encourage you to ask questions via [github issues](https://github.com/chanzuckerberg/cellxgene-census/issues). Alternatively, for quick support you can join the [CZI Science Community](https://czi.co/science-slack) on Slack and join the `#cellxgene-census-users` channel
- Bugs: please submit a [github issue](https://github.com/chanzuckerberg/cellxgene-census/issues). 
- Security issues: if you believe you have found a security issue, in lieu of filing an issue please responsibly disclose it by contacting <security@chanzuckerberg.com>.
- You can send any other feedback to <soma@chanzuckerberg.com>


## Coming soon

- We are currently working on creating the tooling necessary to perform data modeling at scale with seamless integration of the Census and [PyTorch](https://pytorch.org/).
- To increase the usability of the Census for research, in 2023 and 2024 we are planning to explore the following areas:
   - Include organism-wide normalized layers.
   - Include Organism-wide embeddings.
   - On-demand information-rich subsampling.

## Projects and tools using Census

If you are interested in listing a project here, please reach out to us at <soma@chanzuckerberg.com>
