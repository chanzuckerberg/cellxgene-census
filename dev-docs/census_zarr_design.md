# Publishing the Census as AnnData-Zarr

**Status:** proposed, ready for review. **Last edited:** 2026-08-03.

**Approval is needed in three areas**, not from one owner:

- **Census schema:** the major version bump and the rewritten encoding half ([section 5](#5-schema-and-release-changes)).
- **Storage and release policy:** the additive amendment, plus the three things [section 10](#10-decisions-requested) asks it to settle. The approver for this document still needs identifying.
- **Python client:** the v2 surface and its migration path ([section 6](#6-the-python-api)).

This document specifies how the [CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/) is stored and read after it moves off TileDB, stating each decision with the figure behind it. The measurements, their methods and their limits are in [census_anndata_zarr_migration_design.md](./census_anndata_zarr_migration_design.md), which remains the source of truth for evidence.

<!-- TODO: review and sanitize the evidence doc before this goes out for review. It is 1,864 lines and still
     carries supersession passages, process narration and TBD stubs. Either clean it up in place or stage a
     shorter version for reviewers; keep this document's section cross-references in sync if it is renumbered. -->

---

## Summary

Biohub is retiring TileDB across the cellxgene stack. The Census is published today as a [TileDB-SOMA](https://github.com/single-cell-data/TileDB-SOMA) collection and read through a Python client built directly on `tiledbsoma`. This proposes republishing it in the [scverse](https://scverse.org) formats (**[AnnData](https://anndata.readthedocs.io/)**, stored in **[Zarr](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)**) and replacing the client with a version that reads them.

**What changes.** The storage format, and the client API that reads it. The current API is written in terms of SOMA and TileDB types, so it cannot be carried over to a different format unchanged, and the client gets a new major version.

**What does not change.** The data. The same datasets on the same inclusion criteria, harmonized the same way. A build from the new writer must match a TileDB build from the same inputs, cell for cell and value for value, verified before release ([section 7](#7-what-existing-users-are-promised)).

**What existing users get.** Everything already published stays published: pinned versions keep resolving and reading, and the five-year availability commitment on LTS releases is honoured. Data that external groups computed against the old format can be re-joined to the new one through a published mapping. Whether new releases also carry a TileDB corpus, or the move to Zarr is announced as a breaking change, is left open here for the team to decide ([section 7](#7-what-existing-users-are-promised)).

**Performance.** The current format's cost for a selective query is dominated by finding the matching cells, and that cost grows with the size of the corpus rather than the size of the answer, the property that gets worse every release. The proposed layout targets that specific cost, and holds or improves on the current format elsewhere.

**Six things need a decision** before implementation starts; they are listed at the end. The first is whether this becomes a major version of the Census schema.

---

## Terms

Short definitions for terms used throughout.

| Term | Meaning |
|---|---|
| **AnnData** | The standard Python container for single-cell data: a cell-by-gene matrix plus tables describing the rows and columns. |
| **Zarr** | A format for storing large arrays as many small compressed pieces, designed to be read directly from cloud storage without downloading the whole thing. |
| **obs** | The cell table. One row per cell, ~28 columns: tissue, cell type, disease, assay, and so on. Predicates ("all lung cells") are answered here. |
| **var** | The gene table. One row per gene. |
| **X** | The expression matrix itself: cells × genes. Almost all of the data volume. |
| **chunk** | The smallest piece of a Zarr array that can be fetched. Reading one value costs the whole chunk containing it. |
| **shard** | A single stored object holding many chunks, with an index so a reader can fetch just the chunks it wants. Lets a layout have small chunks without having a huge number of objects. |
| **CSR** | The standard way to store a sparse matrix: keep only the non-zero values, row by row. Most cells express only ~1% of genes, so this is far smaller than storing every zero. |
| **band** | A fixed slice of the gene axis, stored as its own matrix. Splitting the genes into 16 bands means a query for a few genes reads only the bands holding them, instead of the whole matrix. |
| **predicate** | The condition a query filters on: "cell type is X", "at least 500 genes detected". |
| **SOMA / TileDB** | The current format and the storage engine underneath it. Being retired. |
| **LTS release** | A long-term-support Census build, guaranteed available for at least five years. |
| **dataset** | One contributed study, curated by CELLxGENE Discover. The Census concatenates 1,845 of them. |
| **harmonization** | Combining separately curated datasets into one: a single shared gene list, one concatenated cell table, and uniformly computed per-cell statistics and normalization. Standardizing labels to ontology terms happens upstream, in CELLxGENE Discover curation. |

---

## 1. Why this change

The tooling direction across Biohub's cellxgene work is scverse, which means AnnData and Zarr. The Census is currently a TileDB-SOMA collection read through a client that wraps `tiledbsoma` directly, so both the stored corpus and its API are in scope.

Two alternatives are out of scope. Keeping the [SOMA](https://github.com/single-cell-data/SOMA) API over a different storage engine would mean building that engine first; there is no SOMA-over-Zarr implementation to adopt. And the format cannot be swapped quietly under the current API: six of the eleven public functions in the client's top-level namespace name SOMA types, take TileDB settings, or return SOMA objects. The `experimental` namespace adds more ([section 6](#6-the-python-api)).

### 1.1 What this has to achieve

The bar for everything that follows.

1. **A narrow question should cost in proportion to its answer, not to the size of the corpus.** Today a one-column filter finds 565 cells for 0.4 MB; adding a second column stops the engine skipping data, and the filters in the client's own documentation cost about 9× as much as a result. That is the problem being solved.
2. **Reading does not get worse anywhere else.** Reading whole cells and reading whole columns stay at or below what they cost today.
3. **Ordinary tools can read it.** Anyone with AnnData and Zarr can open the corpus. The extra index described in [section 3](#3-storage-layout) is optional: a reader that understands it fetches less data to answer the same question, and a reader that ignores it reads the columns straight through and gets identical results.
4. **The answers are the same as today's**, cell for cell and value for value, tested and verified ([section 7](#7-what-existing-users-are-promised)).
5. **No existing user loses access.** Pinned clients and pinned releases keep working, and nothing already published is moved or withdrawn ([section 7](#7-what-existing-users-are-promised)).
6. **Work computed against the old corpus re-attaches to the right cells.** Embeddings and annotations keyed on the old row numbers either land on the same cells they were computed on, or fail with an error, never attaching to the wrong cells quietly ([section 4.3](#43-re-joining-existing-work)).

Stored size is not on this list. It changes, and the measurements point down rather than up ([section 3.1](#31-the-decisions)), but no target is set, because the corpus-scale figure is not yet known.

---

## 2. What exists today

The published corpus, as of the `2025-11-08` LTS release.

**Scale.** 1,845 datasets, **217.8 million observations**, of which 125.5 million are unique cells: the difference is repeat appearances of the same cell across more than one contributed dataset. Stored, that is **1.4 TiB across 325,695 objects**.

**Organisms and modalities.** Seven in total: five non-spatial (human, mouse, macaque, marmoset, chimpanzee) and two spatial (human, mouse). Human dominates at 162 million observations across both, of which the non-spatial side is 159 million; the three smallest organisms together are 4.3% of the total.

**Structure.** Each of the seven holds the cell table, the gene table, and one or two expression matrices: raw counts always, and a normalized layer for non-spatial data. Alongside them sits a **gene-presence matrix** recording which genes were actually measured in which dataset. The gene table is a union across all datasets, so most genes are absent from most datasets; without the presence matrix, "not measured here" and "measured as zero" are indistinguishable. It must be preserved.

The two spatial ones add four kinds of object with no non-spatial equivalent: scenes, point clouds, images, and a cell-to-scene mapping.

**Release cadence.** The most recent build of any kind is `2025-11-17`, which currently carries the `latest` alias. What the transition should publish depends on the intended cadence ([section 7](#7-what-existing-users-are-promised)).

**Two properties of the current data that change how filters must be written**, both introduced by [schema 2.4.0](../docs/cellxgene_census_schema.md):

- **`disease` and `disease_ontology_term_id` can each hold several values in one field**, separated by ` || `. A filter written as exact string equality returns a subset of the matching cells, and raises nothing.
- **Gene symbols are not unique.** Two different genes can carry the same symbol. Only the [Ensembl](https://www.ensembl.org/) accession (`feature_id`) identifies a gene.

---

## 3. Storage layout

One store per organism, holding the cell table, the gene table, and the expression matrix.

```
census_zarr/<organism>/
│
├── obs/                        cell metadata: where "which cells?" is answered
│   ├── obs_index               per-chunk summary: which values each chunk contains,
│   │                           and min/max for numeric columns. Read first.
│   ├── cell_type_ontology_term_id/   chunks of 8,192 rows, 65,536-row shards
│   ├── tissue_general_ontology_term_id/
│   ├── nnz, is_primary_data, …       same chunking
│   └──                         rows stay in the order datasets were added
│
├── var/                        gene metadata
│
└── X/                          expression: sparse, split into 16 gene bands
    ├── band_000/               genes 0…3,842
    │   ├── data                the non-zero values
    │   ├── indices             which gene each value belongs to, numbered
    │   │                       within the band rather than across all 61,497
    │   └── indptr              where each cell's values start
    ├── band_001/               genes 3,843…7,686
    ├── …
    └── band_015/               genes 57,653…61,496
```

### 3.1 The decisions

Each row was chosen against measured competitors. For the expression matrix those were four encodings of the same matrix (banded CSR, plain CSR, gene-major CSC, and a dense two-dimensional array), each required to return identical values. The full comparison, including what each alternative cost, is [section 10 of the evidence document](./census_anndata_zarr_migration_design.md#10-layout-comparison).

| Decision | Choice | Why, in one figure |
|---|---|---|
| **Expression matrix** | Sparse, split into **16 gene bands**, each stored as CSR | Reading one gene costs **0.51 MB** instead of **98.7 MB**. Plain CSR has no gene dimension, so it reads the entire matrix to return one column. Whole-cell reads stay within 0.4% of plain CSR, and this is also the smallest of the four encodings tried. |
| **Where matrix chunks start** | Pad each dataset's values so it begins on a chunk boundary and no chunk mixes two datasets | Fewer bytes on every query measured: **17–35%** at the scale tested, less for datasets larger than a chunk. The padding costs essentially nothing to store: it compresses away, changing stored size by **−0.1%**. |
| **Cell-table chunking** | Uniform **8,192 rows** | AnnData's default writes each column as a single chunk, so nothing can be skipped. 8,192 issues **~1.7× fewer requests** than a finer 2,048. |
| **Sharding** | Yes, **65,536-row** shards | Eight chunks per shard, so **8× fewer stored objects**, with no meaningful change in bytes. The finer 2,048-row chunks measured 28×, but those are rejected above. |
| **Cell-table index** | A per-chunk summary stored beside the data, used **adaptively** | The index is **0.012–0.15 MB** and is read first, so the reader knows how much it would skip before fetching anything. It prunes when that is under ~1/20 of chunks and reads normally otherwise. |
| **Row order** | Unchanged: the order datasets were added | Already within **1.2–1.7×** of optimal, because a cell type concentrates in the few datasets that assayed it. Reordering by cell type gained 6% on one query and cost 210% on another. |
| **Store organization** | **One store per organism**, not one per dataset | Per-dataset stores cost **114 extra requests per store opened**, roughly 211,000 for a query spanning the corpus. |

**Why padding rather than exact boundaries.** Letting each dataset's chunk be exactly as wide as the dataset is not buildable today: Zarr's variable-width chunks are experimental in version 3.2.1, off by default, and cannot be combined with sharding, and sharding is what buys the object count below. Padding achieves the same result with ordinary chunks, so the layout depends on no experimental feature.

**What it costs to store.**

*Object count.* Sharding makes object count grow with data volume rather than with the number of pieces the data is cut into: a shard holds more chunks instead of the store holding more objects. No spike store exceeded about 700 objects at any scale tested, where the unsharded equivalent would hold eight times as many. Today's corpus is 325,695 objects. This is a property of the layout rather than of the sample, so it carries to full scale.

*Bytes per cell measured lower, but treat that as a floor.* A spike store of 64,798 cells worked out to **3.3 kB per cell** against today's **6.9 kB** corpus-wide. The cells in that sample carry fewer non-zero values than the corpus average, so the real figure will land higher than 3.3 kB, and how much higher is unknown until a full build exists. Nothing measured suggests the Zarr corpus is larger than the TileDB one, which is what bounds the cost of publishing both at once ([section 7](#7-what-existing-users-are-promised)).

### 3.2 Why selective queries cost less

The mechanism is narrower than "Zarr is faster".

**The problem being solved.** Against the current format, finding cells is cheap when a query names one column and expensive as soon as it names two. Locating 565 cells among 159 million by dataset costs **0.4 MB and 9 requests**. Adding a second condition, even one that cannot change the answer, stops the engine skipping data, so it reads the whole column. Depending on which column, that costs between 2.5× and 843× more. The filters in the client's own documentation land at the low end, **9.2×**, or 3.4 MB.

This cost tracks **the size of the corpus, not the size of the answer**. Asking for ten times as many cells is nearly free; asking at all is expensive, and gets more expensive with every release.

**How the layout answers it.** The per-chunk summary lets a reader decide which pieces of the cell table can possibly contain a match and skip the rest, and it can do that on any number of columns at once. Measured against the real corpus, a query scoped to one dataset needs **2.6%** of the cell table's chunks, and a typical cell type needs **0.3%**.

The reader consults the summary first, and only then reads:

```
      query: cell_type == X and nnz >= 500
                    │
                    ▼
      read the per-chunk summary        (one small object)
                    │
                    ▼
      which chunks could match?  ──── most are excluded ──▶ never fetched
                    │
             (if few enough survive, fetch just those;
              if most survive, read normally, cheaper)
                    │
                    ▼
      evaluate the condition on what was read → matching cells
                    │
                    ▼
      which gene bands does the query need?
             all genes ──▶ read those cells from every band
             some genes ─▶ read them from only those bands
```

**One limitation.** A threshold on a numeric column ("cells with at least 500 genes detected") cannot be skipped this way, because almost every chunk contains some cells above and some below any threshold. In practice that particular condition also matches 94.5% of all cells, so there is little to skip regardless. It could be made skippable by storing the answer as a ready-made true/false column, which is declined deliberately: that condition belongs to one consumer, and the published corpus should not grow a column for each of them.

### 3.3 Open questions

- **How spatial data is stored.** Scenes, point clouds, images and the cell-to-scene mapping have no AnnData equivalent. [spatialdata](https://spatialdata.scverse.org/) is the scverse counterpart and is already a dependency of both the builder and the client, but the mapping to it is not specified.
- **Whether to reorder genes.** Splitting genes into bands only helps when a query asks for genes that sit near each other. Real queries ask for scattered marker panels, so the benefit is currently limited to single-gene lookups. Grouping co-queried genes together would extend it; no good grouping key is known.
- **The number of bands**, fixed at 16 and not tuned. It trades gene selectivity against object count.

---

## 4. How cells and genes are identified

### 4.1 The problem with today's identifier

Cells are currently addressed by `soma_joinid`, which is simply a cell's row number at build time. Nothing documents it as stable across releases, and it shifts whenever datasets are added, removed or reordered. The ecosystem already treats it as release-scoped: the client refuses to load a stored embedding whose recorded Census version does not match the one being opened.

### 4.2 What replaces it

**A cell is identified by the pair `(dataset_id, observation_joinid)`**: which dataset it came from, and its identity within that dataset. Both columns already exist and are required by the Census schema, so nothing new is computed. The pair is needed because `observation_joinid` is not unique on its own; at least one organism has collisions across its datasets.

Verified on 68 datasets and 6.0 million cells with no violation, sampling 40 of human's 1,086 and leaving mouse unchecked. The property is one the CELLxGENE dataset schema already guarantees, so a violation would be an upstream data bug rather than a flaw here, but the builder should assert it on every release, which costs one pass over a column it already holds.

**A gene is identified by `feature_id`**, the Ensembl accession, never by symbol. Symbols are not unique ([section 2](#2-what-exists-today)).

A cell's identity is stable for as long as its dataset is not re-curated. Re-curation mints a new dataset version upstream, and this design inherits whatever guarantee CELLxGENE Discover makes there.

### 4.3 Re-joining existing work

Groups outside Biohub hold embeddings and annotations keyed on the old row numbers. Every Zarr release therefore also publishes a **mapping table** per organism: old row number → the new pair. It is a projection of two existing columns and is small: 1 MB for the smallest organism, 19 MB for the next.

Two requirements on it: a mapping is only meaningful together with the release it was built from, so one is published per release; and a lookup that cannot be resolved must **fail rather than silently drop rows**. A row number absent from the mapping means that cell is not in this release, which is a real answer and must be reported as one.

This is written for releases that carry both formats, since the old row numbers are assigned by the TileDB build. If new releases ship Zarr only (decision 5), the mapping has to be derived from the last TileDB release instead, which costs nothing extra: both columns are already published there.

---

## 5. Schema and release changes

Two governing documents currently require SOMA and must change: the [Census schema](../docs/cellxgene_census_schema.md) and the [storage and release policy](./cellxgene_census_storage_and_release_policy.md). The public docsite carries the same SOMA-based definition of a release and needs the same amendment.

**The Census schema needs a major version, 3.0.0.** It divides in two. The half describing *what data is included* (organisms, assays, sample types) is untouched. The half describing *how data is encoded* specifies every object as a SOMA type, roughly sixty references, and is rewritten. The schema's own rule makes any change incompatible with the SOMA API a major bump.

**The release policy must admit a second format.** Three clauses change: a build is defined by its content and schema rather than by its container; releases may carry a `zarr/` folder beside the existing `soma/` and `h5ads/`; and the release index records a third location under the key `zarr`, reusing the existing location structure unchanged.

**The change must be additive.** The `soma` entry cannot be removed or renamed while any supported TileDB release exists, for two independent reasons: already-released clients require that key and fail without it, and users who pinned a build expect it to keep working. This holds however [section 7](#7-what-existing-users-are-promised)'s open choice is settled. It is a statement about the releases already in the index, not about what future ones contain. If future releases do carry both formats, one release tag holds both locations and clients pick the one they understand. Removing the old entry is a separate decision, governed by the five-year support window.

**Two things the amendment should fix while it is open.**

*Reconcile the policy with reality.* It has drifted in five places, among them a filename, the storage bucket, and a required alias that has never existed on any release.

*Make the requirements checkable.* That alias has been required since 2023, is unmet by all six LTS releases, and nothing consumes it. So the amendment should come with a build-time check that blocks a release not satisfying it.

Worth knowing when deciding who signs this off: the policy file has exactly one commit in this repository's history, a 2024-05-28 link fix, and no substantive edit ever.

**One new requirement.** The per-chunk index in [section 3](#3-storage-layout) has no home in either the Zarr or the AnnData specification. It sits beside the data as an extension. The schema must define where it lives and what it contains, or the index is usable only by Biohub's own client. The schema should also state that the published layout depends on no experimental Zarr feature, so that a future Zarr release cannot invalidate a published corpus.

---

## 6. The Python API

A new major version of `cellxgene-census`, or a separate package (decision 4). Either way the shape of the current API is kept (open a handle, pass it to accessors), so migrating existing code is mostly mechanical.

The table below covers the eleven functions in the top-level namespace.

| Today | Proposed | What changes |
|---|---|---|
| `open_soma(...) → soma.Collection` | `open_census(...) → Census` | Name and return type no longer mention SOMA; TileDB configuration arguments disappear. The handle is opaque and exposes no storage internals. |
| `get_default_soma_context(...)` | removed | Exists only to hand callers a TileDB object. |
| `get_anndata(census, …)` | same shape | Already returns AnnData. The handle, filter and coordinate arguments change. |
| `get_obs`, `get_var` | same shape | Already return pandas tables. |
| `get_presence_matrix` | same shape | Must survive, or measured-versus-zero is lost ([section 2](#2-what-exists-today)). |
| (none) | `get_anndata_lazy(…)` | New. Returns a result that is described but not yet loaded, so a query spanning an entire organism, up to 159 million cells, can be expressed without materialising it. A separate function rather than a flag, because the return value differs in kind. |
| `download_source_h5ad`, release-directory functions | unchanged | Source files and the release index are unaffected. |

**Selecting cells and genes.** Where the current API takes row numbers, the new one takes the identifiers of [section 4](#4-how-cells-and-genes-are-identified): `(dataset_id, observation_joinid)` pairs for cells, Ensembl accessions for genes. A positional fast path is kept for callers passing back offsets they just read from a previous result, documented as valid only within one release.

**Filters.** Today's filter strings, such as `tissue_general == 'lung'`, keep working, because that is what every existing notebook passes. The new client also takes a pandas-style form, which is what its reader works in natively. The two must select exactly the same cells, and that is tested directly.

The translator has one trap to avoid. `disease` and `disease_ontology_term_id` can each hold several values at once ([section 2](#2-what-exists-today)), so a filter on them is asking whether a value is among those present, not whether the whole field equals it. Treating them as plain strings quietly returns too few cells. Which columns behave this way should come from the schema rather than a hardcoded list, so the same bug does not reappear when another column becomes multi-valued.

**Open questions:** the accessor for spatial objects, which depends on how they are stored ([section 3.3](#33-open-questions)); whether matching a gene by a non-unique symbol is an error or a documented widening; and the `experimental` namespace, which is also SOMA-coupled and is not scoped here, including `get_embedding`, which is keyed on the retired identifier.

---

## 7. What existing users are promised

### 7.1 What stays available

Nothing already published is moved, rewritten or withdrawn, and a user who pinned a client version and a release keeps resolving the same release and reading the same bytes. The five-year availability commitment on LTS releases is honoured, though it is recorded on the public docsite rather than in the release policy. Retiring the TileDB corpus is a later decision, gated on that support window rather than on the new client's readiness.

### 7.2 Whether new releases carry both formats

This is an open choice (decision 5). Two shapes are available:

- *Publish both.* Old clients keep working against new data, and the transition is invisible to anyone who does not want it. The cost is running and validating two writers for as long as it lasts.
- *Publish Zarr only, and announce it.* Users who want new data move to the v2 client on a published timeline; those who do not stay on the last TileDB release, still available under its support window. The cost is that the move becomes mandatory.

The choice depends on the release cadence ([section 2](#2-what-exists-today)), not on anything in this design: both shapes work with the layout, the identity model and the API. It does change the release-policy amendment, since publishing both needs one release tag to carry two locations.

### 7.3 What gates the first Zarr release

**It must match the current one.** The same queries run through both stacks, and the results must agree on which cells, which genes, the values, and the metadata columns. A faster stack that fails this fails outright. Because it is a comparison, a handful of datasets is enough to catch an encoding mistake, so it does not wait on a full-scale corpus. Section 11 of the evidence document defines the comparison and the query set.

**No externally contributed data silently mis-joins.** Work keyed on old row numbers re-joins through the mapping in [section 4.3](#43-re-joining-existing-work), and a join that cannot be performed fails rather than returning the wrong cells. This must be demonstrated against a real contributed embedding before the first Zarr release.

### 7.4 The R client

**It keeps reading what it reads today**, since it reads the TileDB corpus and those releases stay published. Whether it also sees new data depends on decision 5.

A route to the new format exists and is recent. [anndataR](https://bioconductor.org/packages/anndataR/) reads AnnData-Zarr in its current Bioconductor release and hands its Zarr work to [Rarr](https://bioconductor.org/packages/Rarr/), where sharding reached the development branch in June 2026, read-only, and is expected in a release around October 2026. So R parity is separate work rather than a missing reader. Worth confirming early: that `anndataR` over a development `Rarr` really does read a sharded store, since neither project appears to test that. (State of the R ecosystem as of August 2026.)

Until then R users cannot reach a Zarr-only release. Under decision 5's second shape they stay on the last TileDB release, keeping what they have but seeing no new data, and lose access only if that corpus is retired first.

### 7.5 Deprecation

**Deprecation follows existing practice** in this package: an argument that is being replaced keeps working, warns, and names its replacement; supplying both forms is an error.

---

## 8. Risks

Collected here because several would otherwise be met only in passing.

### 8.1 What the measurements do not cover

**Nothing has been measured at full scale.** A complete Zarr corpus does not exist until the first full build, so every corpus-scale figure here is an extrapolation. Each is stated with the measurement it came from, so it can be rechecked rather than re-argued, and the full build is validated before the first Zarr release.

**Row order has the thinnest support.** The penalty for reordering was measured on a single matrix of 64,798 cells and extended to corpus scale by reasoning rather than by a number. It is cheap to revisit, being the decision to change nothing, and the case that would reopen it is specific: a cell type scattered across hundreds of datasets, which was not tested.

**The index's cut-off point was measured on local disk, not cloud storage.** It balances requests against bytes, and only their relative cost fixes where the line belongs. Correctness does not depend on it: a line in the wrong place costs efficiency, never right answers.

**The identity model is verified on 3.7% of datasets.** The pair `(dataset_id, observation_joinid)` was checked on 68 of 1,845, sampling 40 of human's 1,086 and skipping mouse, because reading all 159 million human pairs at once holds roughly 10 GB of strings. Uniqueness is guaranteed upstream by the CELLxGENE dataset schema, so a violation would be an upstream data bug, and asserting it in the builder costs one pass over a column already held. Until that assertion exists, the coverage is a sample.

### 8.2 What depends on work outside this design

**The build's memory and runtime envelope is unknown.** The spike held whole matrices in memory, which does not scale to an organism, so the builder must write incrementally. The layout cooperates: each gene band is written independently, and because every dataset starts on a chunk boundary, datasets can be appended one at a time without rewriting a neighbour's chunk.

**R support depends on Zarr tooling that is only now arriving.** The reader exists, but sharding is in a development branch expected in a Bioconductor release around October 2026, and nobody has confirmed the combination works end to end ([section 7](#7-what-existing-users-are-promised)). Until then R users cannot reach a Zarr-only release, and would lose access altogether only if the TileDB releases were retired first.

**Gene banding is half-realised.** It helps single-gene lookups and little else until genes are reordered ([section 3.3](#33-open-questions)). That costs nothing to carry, since banded storage is also the smallest encoding measured.

---

## 9. Out of scope

- **What the Census contains.** Dataset inclusion criteria, harmonization, computed statistics, the presence matrix, and upstream curation are all unchanged.
- **The build pipeline's structure.** Its stages keep their order. What changes is the component that writes output, plus the removal of two TileDB-specific finalization steps with no Zarr equivalent.
- **Retiring the TileDB corpus and the current client**, which is governed by the support commitment.
- **The R client's own implementation.**
- **CZ CELLxGENE Discover portal integration** with the new client.
- **Regenerating hosted embeddings.** The mapping keeps existing ones joinable; retraining models is not in scope.
- **The Where's My Gene service's separate TileDB use**, which is independent of the Census corpus.

---

## 10. Decisions requested

1. **Does this become Census schema 3.0.0?** [Section 5](#5-schema-and-release-changes) argues it must. If so: does 3.0.0 branch from the current 2.4.0 as published, or wait for 2.4.0 to be finalised, which is still marked as drafting.
2. **Is the release-policy change additive as described**, with the `soma` entry retained for the life of the TileDB support window? Three things the same amendment should settle rather than leave open: how the unmet `V<#>` alias requirement is resolved, whether the amendment ships with a build-time conformance check, and that the five-year availability commitment currently lives on the public docsite rather than in the policy itself.
3. **How long is the identifier mapping published after the transition?** Keeping it indefinitely makes old externally held data re-joinable forever; dropping it gives the transition window meaning.
4. **Does the new client ship as a major version of `cellxgene-census` or as a separate package?**
5. **Do new releases carry both formats, or does the move to Zarr ship as an announced breaking change?** [Section 7](#7-what-existing-users-are-promised) sets out both shapes and does not pick one, because the answer depends on the intended release cadence rather than on anything in this design. It is the decision that most affects what existing users experience, and it sets how much of the builder has to keep working in two modes.

6. **Is `soma_joinid` intended to be stable across releases?** [Section 4](#4-how-cells-and-genes-are-identified) argues the new identity is needed because nothing documents the old one as stable. It is stable in practice today, which is why people join on it, but no guarantee appears to be written down. If the answer is that stability was always intended, section 4's premise needs revisiting.

Smaller, but they should not be left to the implementation:

- Whether the published corpus keeps the full set of category labels or only those present in a result. The two formats differ today and neither raises.
- Whether `var.feature_name` is published as a string or a categorical. Like the point above, this is an AnnData writer default today rather than a design choice, and it changes what a caller sees.
- Whether a gene symbol matching several genes is an error or a documented widening.
- Confirmation of the adaptive index rule in [section 3.1](#31-the-decisions), since it is a behaviour of the published read path rather than an internal detail.

---

## Evidence

Every figure in this document is measured. The measurements, their methods, and their limits are in **[census_anndata_zarr_migration_design.md](./census_anndata_zarr_migration_design.md)**, which is the source of truth for evidence and is organised as: [read-path measurements against the current format](./census_anndata_zarr_migration_design.md#9-read-path-feasibility-evidence), [layout comparisons](./census_anndata_zarr_migration_design.md#10-layout-comparison), [validation methodology](./census_anndata_zarr_migration_design.md#11-validation-methodology), and [build-cost bounds](./census_anndata_zarr_migration_design.md#13-build-resource-envelope). Raw results are in [tools/zarr_layout_spike/results/](../tools/zarr_layout_spike/results/).

One thing to carry into any reading of the numbers: no timing claims are made anywhere in this document. Read timings taken from outside the corpus's AWS region varied by more than thirtyfold on queries that their byte counts prove did identical work, so everything here is expressed in bytes transferred and requests issued, which are exactly reproducible. The other limit, that nothing has been measured at full scale, is section 8.
