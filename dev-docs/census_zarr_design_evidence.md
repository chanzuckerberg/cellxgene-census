# Census AnnData-Zarr: measurements and methods

The evidence behind [census_zarr_design.md](./census_zarr_design.md), which is the specification and states
the decisions. This document is the source of truth for every figure quoted there: what was measured, how, and
what each measurement does not establish.

Raw results are in `tools/zarr_layout_spike/results/*.csv`. Sections are numbered 9 onward because they keep the
numbers the specification cites. References written `spec §N` point into that document.

Two terms recur. A **rung** is one scale point of the measurement series, named for how many source datasets it
holds: rung 01 is one dataset, rung 25 is twenty-five. The **ladder** is the series of rungs, run end to end so a
figure can be read as a curve against dataset count rather than as a single number.

**Last edited:** 2026-08-03.

---


## 9. Read-path feasibility evidence

**Delivered.** 28 measured queries across rungs 01, 05 and 25 in two experiments, plus an `obs`-predicate probe.
Every non-empty query agreed on cells, genes and `X` — including `normalized` within tolerance — with the only
differences being the axis-metadata encoding class of 9.5. Per-query bytes and requests are in
`tools/zarr_layout_spike/results/bench.csv`.

The read-path question resolves to a mechanism rather than a ratio:

> A single-column predicate locates 565 rows in 158,982,719 for **0.4 MB and 9 requests**. Adding one clause that
> cannot change the selection defeats TileDB's fragment pruning, after which the `obs` axis is scanned. Every other
> cost result in this section follows from that.

9.3 is the subsection to read if you read only one.

**Outstanding**, none of it blocking the layout decision: rung 100 and the extrapolation to 1,845 datasets, which would
confirm rather than discover, since 9.2 gives the scaling law; in-region wall clock, deliberately not attempted
(10.1); and cold versus warm, not separated — every figure here is a cold read.

### 9.1 What the ladder compares, and what it cannot

**The two stacks do not read the same corpus, so their byte columns are not a head-to-head.** The SOMA side
answers every query from the whole published corpus; the Zarr side answers from a store holding only the rung's
datasets, because a full-corpus Zarr copy is 1.4 TiB and does not exist. The ratio between the two columns is
dominated by corpus scope, not by layout. What the run legitimately supports is the Zarr scaling curve across
rungs, the SOMA cost of retrieving a small subset from the full corpus, and parity at every rung.
Layout-versus-layout at equal scope is Section 10, where every candidate holds the same data.

**Size-ascending rungs select secondary data, which silently disables the WMG query.** Of the 100 smallest
datasets in the release, 73 carry no primary cells, and rungs 01 and 05 carry none between them, because secondary
datasets are typically subsets or re-annotations of larger ones. Since the WMG filter requires
`is_primary_data == True` it selects zero cells below rung 25, and the rungs also skew shallow: median `nnz` per
cell at rung 01 is 292 against the filter's floor of 500. A defect in the manifest selection criteria (11.5), not
in the queries; fixing it invalidates measurements already taken.

**Two metric floors.** Amplification divides compressed bytes read by an uncompressed payload
(`bytes_wanted` is `nnz * 12`), so below about 3x it measures compression rather than over-fetch — two rung-25
queries report 0.5x and 0.6x. Never quote it without the payload beside it. And `bench.py` re-probes filter values
from each subset, which is what stops the queries selecting nothing, so each rung binds a *different* cell type.
The rungs are the same query *shape* at growing scale, not the same query: request counts and byte totals compare
across rungs, anything payload-normalised does not.

### 9.2 The current stack's cost is fixed per query and proportional to the experiment

Seven queries at three rungs against the published corpus, `homo_sapiens`. Every rung reads the real corpus and
only the manifest scope in the predicate changes, so this carries no scope confound.

| Query | Cells, 01 → 05 → 25 | SOMA MB, 01 → 05 → 25 | SOMA requests |
|---|---|---|---|
| `cell_type_slice` | 391 → 3,799 → 11,470 | 150.1 → 154.7 → 163.6 | 294 → 294 → 294 |
| `dataset_slice` | 565 → 3,799 → 15,502 | 99.1 → 103.7 → 107.8 | 287 → 287 → 287 |
| `tissue_slice` | 565 → 3,799 → 30,714 | 101.6 → 106.2 → 171.5 | 287 → 287 → 291 |
| `both_layers` | 391 → 3,799 → 11,470 | 248.8 → 259.5 → 277.1 | 335 → 335 → 335 |
| `multivalued_disease` | 565 → 3,158 → 35,808 | 103.2 → 104.7 → 173.2 | 291 → 291 → 295 |
| `cell_type_gene_subset` | 391 → 3,799 → 11,470 | 147.3 → 153.6 → 163.0 | 294 → 294 → 294 |
| `wmg_production` | 0 → 0 → 2,728 | 625.9 → 625.9 → 451.9 | 353 → 353 → 331 |

**Request count is a constant of the query shape**, identical to the unit at rungs 01 and 05 and almost invariant
at rung 25 across a 100x change in result size: `cell_type_slice` issued exactly 294 requests at 391, 3,799 and
11,470 cells. For the metric 10.1 identifies as dominant, TileDB's does not respond to how much data is returned.

**Cost is not monotonic in result size.** `wmg_production` returned the same 625,879,749 bytes at rungs 01 and 05 —
the same number to the byte, with five times as many datasets in scope — then **451.9 MB for 2,728 cells** at rung
25, 174 MB less to do strictly more. A zero-cell query is the most expensive on its rung: with nothing selected
there is no `X` to fetch, so 625.9 MB in 353 requests is `obs`-axis predicate evaluation and nothing else, at 4x
the cost of the queries that return data. Bytes do eventually move — at rung 25 `tissue_slice` rose about 65%,
where rungs 01 → 05 gave 3--6% — but at 11x the cells for 1.6x the bytes the crossover is a knee, not a corner.

**The fixed cost is proportional to the experiment.** Rung 25 contains one *Macaca mulatta* dataset alongside its
24 human ones, so the same queries ran against two experiments differing **22.7x** in row count — 158,982,719
against 7,010,229. Cut within the rung, human costs 19.3x to 30.9x the macaque figure; cut instead against human at
rung 05, where result sizes are close at 3,799 against 3,190 cells, 18.6x to 28.3x. Both bracket 22.7x, and the
cleanest case is `wmg_production` matched on cells, where **both sides select exactly zero** and the cost still
differs by 28.3x. Normalised, human reads 0.6 bytes per `obs` row for a single-column predicate and 3.9 for the WMG
predicate; macaque 0.8 and 3.2. So:

> Bytes fetched are set by the number of rows in the `obs` axis and by which columns the predicate touches. They
> are very nearly independent of how many cells the query selects.

**This is the one property that gets worse as the Census grows.** A per-query cost proportional to corpus size
means every selective read pays for every cell ever added. The corpus is 218M observations across 1,845 datasets
today (spec §2) and grows with each release; no chunk or shard tuning addresses that, because the `obs` axis is being
scanned. 9.3 pins down why.

**Where the bytes go on the `X` side.** Instrumented with `tiledbsoma_stats`, a 167-cell read of `pan_troglodytes`
— 0.11% of cells, 0.034% of values — wants 1.61 MB and fetches **510.67 MB**, an amplification of **317x** in 312
requests. The tile geometry accounts for it exactly: `X` is stored in **2048 x 2048** tiles under Zstd
(`CENSUS_DEFAULT_X_LAYERS_PLATFORM_CONFIG`) and a compressed tile is atomic. The experiment is 78 row-bands by 12
column-bands; 167 scattered cells land in roughly 69 row-bands, and because the query reads every gene each touched
band pulls all 12 column-bands — roughly 828 of 936 tiles, about 88% of the array, consistent with 88% of a ~580 MB
compressed array. Request count is not the problem: 312 requests for 510 MB is 1.8 MB each, which is good
coalescing. Read 317x with its payload beside it; since the numerator is a fixed cost the ratio falls by whatever
factor the payload grows, so 317x is what a 167-cell read costs, not what the corpus costs.

Bounding these claims: two experiments, seven queries, one client, one laptop. The metric is deterministic counters,
which reproduce exactly (10.1), so the ratios carry no network variance. The macaque subset is one dataset against
human's 24, so their predicates differ in the length of the `dataset_id` scope clause — measured to contribute
nothing, which is what makes either cut usable. This describes TileDB on the published corpus; that a full-axis
scan is avoidable in principle is not evidence that the mandated stack avoids it.

### 9.3 The mechanism: the second predicate column defeats fragment pruning

`obs`-only reads against `homo_sapiens`, projecting `soma_joinid` alone so the returned payload is identical
throughout, varying nothing but the predicate.

| Predicate | Rows | Bytes | Requests |
|---|---:|---:|---:|
| `dataset_id in ['d7476…']` | 565 | **0.4 MB** | 9 |
| the same **+ `nnz >= 0`** | 565 | 258.2 MB | 26 |
| the same **+ two further always-true clauses** | 565 | **259.9 MB** | 50 |
| the same **+ `cell_type_ontology_term_id == 'CL:0000128'`** | 0 | 51.4 MB | 11 |
| the same **+ the WMG clauses** | 0 | **312.2 MB** | 58 |
| unscoped `nnz >= 500` | 150,300,999 | 259.8 MB | 33 |
| unscoped `cell_type_ontology_term_id == 'CL:0000128'` | 8,105,477 | 53.1 MB | 89 |

**A single-column predicate on a dictionary-encoded column is nearly free**: 0.4 MB and 9 requests to locate 565
rows in 158,982,719, because TileDB prunes to the fragments that can contain the value. The fixed cost of 9.2 is
not inherent to selective reads against this corpus — it has a trigger.

**The trigger is the second column, and it is not a per-column cost.** Adding `nnz >= 0` — a clause that cannot
change the selection, and the same 565 rows come back — takes the read from 0.4 MB to 258.2 MB. Adding two
*further* always-true clauses brings it to 259.9 MB, 1.007x rather than 3x. So the second column does not add its
own scan: it **defeats the pruning**, after which the axis is read anyway and further columns cost only their own
compressed size.

Per-column full-axis scan cost varies an order of magnitude with compressibility: `nnz`, a wide-range integer, is
259.8 MB (1.6 bytes/row) against dictionary-encoded `cell_type_ontology_term_id` with 898 levels at 53.1 MB (0.33
bytes/row). The arithmetic closes on the headline figure — WMG references `nnz` and three dictionary columns, and
258 + 51 ≈ 309 MB against 312.2 MB measured. The ladder's larger 626 MB is that scan plus the eight `obs` columns
the query projects, where the probe projects one.

**How large the cliff is depends entirely on which column is named, and the client's documented filters sit at the
bottom of the range.**

| Second clause added to the `dataset_id` baseline | Bytes | vs 0.4 MB |
|---|---:|---:|
| `is_primary_data == True` — client | **0.9 MB** | 2.5x |
| `tissue_general_ontology_term_id == X` — client | **2.8 MB** | 7.7x |
| both client columns together | **3.4 MB** | **9.2x** |
| `cell_type_ontology_term_id == X` (dictionary, 898 levels) | 51.4 MB | 139x |
| `nnz >= 0` (numeric) | 258.2 MB | 697x |
| WMG's four clauses | 312.2 MB | 843x |

**The cliff spans 2.5x to 843x, and documented client usage costs 9.2x — 0.4 MB to 3.4 MB.** The mechanism is the
same at both ends; the consequence is not. The expensive columns are the ones that do not correlate with row order,
at 1.6 and 0.33 bytes per row against roughly 0.01 for `is_primary_data` and `tissue_general` (11.2). **The `nnz`
and WMG figures are the worst case in the corpus, not the workload, and are not the headline.**

Two smaller results:

- **Selecting zero rows is expensive once pruning is defeated**, at 51.4 MB and 312.2 MB for the two zero-row
  predicates, which is why 9.2's zero-cell WMG query is the most expensive on its rung. The cheap experiment misses
  during the ladder run were single-column `dataset_id` predicates whose value was absent from the dictionary, so
  pruning eliminated everything.
- **`is_primary_data in [True, False]` raises** `PyQueryCondition.create_uint8() not found`: the `in` operator is
  unsupported against boolean columns. 6.3 has to accept that as a gap in the dialect it promises compatibility
  with, since a v2 filter that *did* support it would accept filters the oracle rejects.

**What this asks of the layout:**

> Multi-column predicate evaluation must not degrade into a full scan of the `obs` axis. A predicate over k columns
> must cost on the order of the k-column single-value case, not the k-column full-axis scan.

Per-chunk statistics would do it, since a Zarr reader can skip chunks whose statistics exclude the predicate; 10.2
measures how far that gets. It also gives 11.4's threshold a defensible form, and the figure to hold v2 to is the
**client** one: 0.4 MB and 9 requests for a single-column scoped read, 3.4 MB for a documented two-column read. A
threshold set at the 258 MB `nnz` case would clear a bar users never approach — though the layout should still fix
that case, because a numeric-threshold filter is legal, WMG issues one, and 10.2 measures it as the residual
per-chunk min/max cannot prune.

Bounding the claim: one experiment, one client, one session. The always-true clauses are synthetic by design, since
a real clause would change the selection and confound the measurement. The per-column costs are specific to these
columns' encodings.

### 9.4 Sparse `X` in AnnData has no gene axis

`cell_type_gene_subset` asks for 100 of 61,497 genes on the same cells as the all-genes query. Against the Zarr
store it read **29,054,752 bytes in 206 requests** — byte-identical and request-identical to the all-genes query,
not merely similar. On the SOMA side it was within 0.35%, and at rung 01 restricting genes bought 1.9%: 147.3 MB
against 150.1 MB for a payload 149x smaller.

The store explains it. AnnData writes sparse `X` in **CSR as flat one-dimensional arrays**: `data` and `indices` of
51,960,590 elements in 256 chunks of 202,972, plus `indptr`. The chunk grid is over *flattened nonzeros*, so
**there is no gene axis to chunk on**, and no chunk or shard size can make a gene-selective read cheap — the reader
fetches every chunk spanning the selected rows and discards unwanted genes from `indices` in memory.

Three consequences:

1. **TileDB's 2048 x 2048 tiling has a gene dimension and AnnData's sparse encoding does not.** On gene-selective
   reads the current stack is structurally more capable than the mandated target's default encoding. Recovering the
   axis takes a layout decision with real cost — a gene-major (CSC) copy, a dense chunked-2D `X`, or
   per-gene-band arrays — and doing nothing means accepting the regression. 10.3 measures all four.
2. **The column-locality matched pair of 11.2 cannot discriminate** under this encoding, on either stack: it varies
   an axis that does not exist. The pair measured nothing, which is the vacuous-comparison failure class this
   harness was built to surface.
3. **Few objects does not mean few requests.** 256 chunks over two arrays is 512 range requests before `indptr` and
   `obs`, which is why reading 55% of the store's rows took **617 requests** where TileDB took **295** for the same
   cells. Auto-sharding delivered the object-count win — 88 objects for 213.6 MB — but a partial read still issues
   a range request per chunk.

That last figure is the sharpest cross-stack statement the run supports, and requests are not confounded in the
direction that flatters Zarr: for this configuration, holding more data cannot require fewer requests for the same
rows. The configuration is anndata's automatic default, chosen by `auto_shard_zarr_v3` rather than by anyone, so it
measures the default and not Zarr — and 295 requests is the target Section 10 was given.

Store footprint across the rungs — 565 cells at 2.1 MB in 83 objects, 5,716 at 18.3 MB in 84, and 64,798 at 213.6
MB in **88** — is 115x the cells and 100x the bytes for five more objects, because auto-sharding packs each array
into essentially one shard. Against the SOMA baseline of 1.4 TiB across 325,695 objects (spec §2), object count is not
the constraint for a sharded Zarr corpus. Bytes per cell is 3.3 kB here against SOMA's 6.9 kB corpus-wide, but the
ladder's cells are shallower than the corpus mean, so treat 3.3 kB as a floor rather than an extrapolation.

### 9.5 Parity holds, and two encoding differences need a decision

Every non-empty query agreed on cells by source identity, genes by `feature_id`, and `X` on every layer the
experiment carries — `raw` exactly, `normalized` within tolerance. The first cross-stack comparison, a 167-cell
slice of `pan_troglodytes`, matched all 140,587 `raw` values exactly across 167 cells and 23,534 genes.

Every difference the checker reports falls in one class, axis-metadata encoding, and neither half raises. SOMA
carries the experiment's full category dictionary for `obs.cell_type` and its ontology column — 25 cell types, of
which 24 are absent from the Zarr round trip, which retains only the levels present in the slice. And
`var.feature_name` is `str` via SOMA and `category` via Zarr, because AnnData's writer converts string columns to
categorical on write. Both are the failure class this harness exists for: values identical, nothing raised, and the
user still receives something different. A comparison that decategorised before comparing, as the builder's own
validation does, would report a clean pass.

Two decisions this leaves open (spec §10). **Does the published corpus preserve the full category dictionary or only
observed levels?** A caller who filters and then inspects `.cat.categories` gets a different answer under v1 and
v2; both behaviours are defensible, differing silently is not. **Is `var.feature_name` a string or a categorical in
v2?** That is AnnData's writer behaviour rather than a design choice, which is why it is worth settling before it
becomes the contract.

Bounding the claim: the Zarr side was converted **from** the SOMA read, so this tests format fidelity and reader
equivalence, not independent derivation — that arrives when the builder emits Zarr. The missing
categories are partly an artefact of converting a slice rather than a whole experiment, so what the finding
establishes is that category handling differs between the paths and needs specifying, not that the corpus would
lose 24 levels.

## 10. Layout comparison

**Complete for `obs` and for the `X` encoding; one item open.** Every decision, with the number that decided it:

| Axis | Chosen | Rejected, with the number that rejected it |
|---|---|---|
| `X` encoding | **Banded CSR, 16 gene bands** (10.3) | Plain CSR: reads the whole 98.7 MB matrix for one gene. CSC: reads it all for any row query. Dense 2-D: works and costs only 1.38x, but banding matches it more cheaply on today's queries |
| `obs` chunking | **Uniform 8192-row chunks** (10.2, 10.5) | AnnData's default one-chunk-per-column leaves nothing to prune. Dataset-aligned chunks prune 10x better and read 3x more bytes |
| Sharding | **Shard** (10.2) | Unsharded: 28x the objects. Sharding needs adjacent candidate chunks to help requests |
| `obs` index | **Per-chunk statistics sidecar**, applied adaptively below a ~1/20 survivor threshold (10.2) | No index: reads the whole column where a real `dataset_id` scope needs 2.6% of chunks. Always-on: costs ~3 requests per surviving chunk, which loses on broad predicates |
| Row order | **Assignment order, unchanged** (10.4) | Clustering by cell type: gains 6% on its own query's `X` read, costs 210% on another's |
| Store organization | **One store** (10.5) | Per-dataset: 114 extra requests per store opened, ~211,000 for a corpus-wide query |
| `X` chunk edges | **Padded to dataset boundaries** (10.5) | Unaligned: 17--35% more bytes and up to 2.2x the floor. Finer chunks instead: more bytes, 1.8x the requests, a larger store |

**Nothing is open on the layout.** The last item — aligning `X`'s nonzero chunk edges to dataset boundaries, where
the byte saving attributed to per-dataset stores actually lived — is settled in 10.5: padding recovers 17--35% of
the bytes and changes the store by −0.1%.

The two corpus-level metrics this section was asked to record, against the SOMA baseline of 1.4 TiB / 325,695
objects (spec §2): **object count is not a constraint for a sharded layout** — 88 objects held 213.6 MB and 115x more
cells added five (9.4) — and **bytes per cell measured 3.3 kB against SOMA's 6.9 kB**, a floor rather than an
estimate. Corpus size sets the incremental cost of publishing both formats, if that is chosen (spec §7.2); on this floor the Zarr corpus would not
exceed the SOMA one.

### 10.1 How candidates are ranked: counters, not wall clock

**Request count is ranked first, on prior art.** `chanzuckerberg/multimodal-slicing` converted CELLxGENE datasets
to Zarr on S3: reading 16,527 cells originally issued **33,054 S3 requests**, two per cell, and restructuring the
read to fetch contiguous ranges reduced that to **2** — a reported ~150x improvement, taking the read from over ten
minutes to 4--7 seconds. With requests minimised, adding 8 Dask workers produced no speedup at all (3% slower) and
the project raised its parallelism threshold to 50,000 rows, concluding that "I/O dominates over CPU processing
time." Two consequences: bytes and CPU throughput are secondary, which is why object count is a first-class metric
here; and request count is a property of layout *and* access pattern together, since those 2 requests came from
coalescing rather than from storing fewer objects, so every candidate must be read through a path that coalesces or
the comparison measures the reader. That work aligned datasets to a 79,948-gene universe of its own, covers human
only, has no spatial data and is not census-harmonized, so its numbers are directional.

**Wall clock from outside `us-west-2` is unusable, and the counters are exact.** One 167-cell query, run repeatedly
from a laptop in one session, took ~475 s, then ~27 s as the mean of three consecutive runs, then 39 s, 207 s and
349 s — an 18x spread with no cold-then-warm pattern. The control is stronger than the spread: the rung 01 run was
executed twice an hour apart on the same laptop, and all seven queries returned **byte-identical counters** — the
same `VFS.read_byte_num` to the byte, the same request count, the same LIST/HEAD count — while wall clock for those
same reads ranged from 0.17x to 5.34x of the first run, a **31x spread**. `wmg_production` moved 625,879,749 bytes
in 353 requests both times, taking 49.5 s and then 264.1 s; `multivalued_disease` moved 103,180,185 bytes in 291
requests, taking 128.2 s and then 21.4 s. The identical byte counts *prove* the two runs did the same work, so the
time difference cannot be a difference in what was read.

Three consequences, all adopted by 11.4: deterministic counters are the primary metric and wall clock secondary; a
no-regression threshold is stated in requests and bytes, which is possible precisely because those are exact rather
than merely reproducible; and any wall-clock comparison must run **in-region** on an EC2 instance in `us-west-2`,
which is why no timing claim appears in this document.

**A naive reader makes every layout look bad.** A scattered-row `X` read initially reported **36 GB from a 98.7 MB
store**, because it issued one range per contiguous row run and re-fetched the same chunk for every row in it.
Every figure in this section comes from a reader that coalesces by chunk, with a regression test asserting that no
scattered read exceeds a full read.

### 10.2 `obs`: chunk it, shard it, and index the multi-column predicates

The sweep that tests whether a Zarr layout can hold a multi-column predicate near the cost of a single-column one
(9.3). Four axes — chunk size, shard size, row order, chunk statistics on or off — over an `obs` axis scaled to
**5,000,000 rows**, all local, counters exact. 64 measurements in `results/sweep.csv`.

Three preliminaries that shaped it. **AnnData writes each `obs` column as one chunk** — measured `chunks=(64798,)`
on every column of the rung 25 store — so there is no granularity to prune at and chunking `obs` is a precondition,
not a tuning. **Zarr carries no chunk statistics**, unlike Parquet, so they live in a sidecar, which makes the index
an *extension* to the AnnData-Zarr layout needing a defined home in the schema (spec §3.1, §5). And **the rung 25 store
is too small to measure this on**: at 64,798 rows a compressed `obs` column is a few kB and the sidecar is
comparable in size to the data it prunes. The 5M-row axis is built by repeating whole dataset blocks so
`dataset_id` stays contiguous and cell types stay distributed within a dataset as they really are; it is unsuitable
for any claim about corpus content.

**The standing configuration and what it costs.** Assignment order, 8192-row chunks, 65536-row shards — the layout
3.1 specifies. Predicates are a single dictionary column, that column plus `nnz >= 500`, WMG's four clauses, and a
value absent from the axis:

| Predicate | Index on | Index off | Bytes saved | At 20M rows |
|---|---:|---:|---:|---:|
| 1 column | 0.53 MB, 626 reqs | 1.23 MB, 100 reqs | 2.3x | 2.10 / 4.90 MB — 2.3x |
| 2 columns (`+ nnz >= 500`) | **4.19 MB**, 1,005 reqs | 10.38 MB, 181 reqs | 2.5x | 16.75 / 41.49 MB — 2.5x |
| 4 columns (WMG) | 8.88 MB, 4,664 reqs | 10.44 MB, 346 reqs | 1.2x | 35.50 / 41.70 MB — 1.2x |
| absent value | **0.02 MB, 16 reqs** | 1.23 MB, 100 reqs | 61x | 0.03 / 4.90 MB — 157x |

**The index buys bytes and costs requests.** Pruning reads 34% of the chunks for the one- and two-column
predicates, saving 2.3--2.5x on bytes — but under assignment order those surviving chunks are scattered across
shards and each costs its own range request, where an unpruned read fetches whole shards. That is 3.0's second
qualification measured: sharding's request benefit needs adjacent candidates. So the index is worth applying where
pruning is *decisive* rather than partial, and the absent-value row is the clean case, better on both metrics by
61x and 6x. It is not a toy either: 9.3 shows a predicate naming a value the corpus does not hold is a real and
expensive query on the current stack. Which predicates take the index is therefore a reader policy the layout
should state rather than leave to the implementation (Section 14).

**The trade does not improve with scale, and the reason bounds what this sweep can answer.** Repeated at
**20,000,000 rows** (`results/sweep_20m.csv`, same 64 configurations), every absolute figure grows about 4x and
every ratio holds: the byte saving is 2.33x on one column against 2.31x at 5M, 2.48x on two columns against 2.48x,
and the request penalty widens slightly, 6.3x to 7.5x on one column. Only the decisive case improves, from 80x to
157x on bytes.

That flatness is structural rather than a result. The ratio is set by `fraction_read`, the share of chunks the
filtered value occupies, and it came out **0.340 at 20M against 0.339 at 5M** — identical to three decimals,
because the axis is scaled by repeating whole dataset blocks, so a value's chunk occupancy is a property of the
64,798-row block cycle and cannot vary with row count. **This sweep therefore measures the index at one fixed value
distribution**, and that distribution is the thing the answer depends on. So it was measured on the real corpus.

#### What a real predicate actually occupies, measured on the published corpus

`probe_chunk_occupancy.py`, over `census_data/homo_sapiens` in assignment order — 158,982,719 rows, 19,408 chunks of
8192 — asking for each predicate how many chunks contain a match. `soma_joinid` was verified contiguous from 0
first, since positional chunking is only assignment order if it is. Reads are one column at a time and total about
310 MB. Results in `results/chunk_occupancy.csv`; the same run against `macaca_mulatta` is there for contrast.

| Predicate | Rows matched | Of the axis | Chunks touched | Spread |
|---|---:|---:|---:|---:|
| `cell_type == CL:0000540` — the most common | 10,035,322 | 6.3% | **0.124** | 2.0x |
| `cell_type == CL:0002365` — median rank | 16,371 | 0.010% | **0.0030** | 29x |
| `cell_type == CL:0008024` — rarest | 8 | ~0 | **0.0001** (1 chunk) | — |
| `tissue_general == UBERON:0000955` — brain, client-documented | 48,581,856 | 30.6% | 0.328 | 1.1x |
| `dataset_id == 37a17b78…` — the largest | 4,140,453 | 2.6% | **0.0261** | 1.0x |
| `nnz >= 500` — WMG's numeric clause | 150,300,999 | 94.5% | 0.999 | 1.1x |

"Spread" is chunks touched divided by rows matched, so 1.0x is a perfectly contiguous value and higher numbers mean
the rows are scattered. Three things follow.

**The synthetic axis was pessimistic, and by more than a little.** The hardest real dictionary case — the single
most common cell type in the corpus — touches **0.124** of chunks where the synthetic axis reports 0.339, and a
median-frequency cell type touches **0.0030**, a hundredfold better. Sweep figures should be read as an upper bound
on cost, not as the expected case.

**Assignment order is dataset-contiguous, now measured at corpus scale.** `dataset_id` touches 0.0261 of chunks to
return 2.6% of rows, a spread of 1.0x: a dataset's rows occupy exactly the chunks they have to and no others. That
is the property 10.4 relies on when it keeps assignment order, and it had not previously been checked above 64,798
rows.

**The `nnz` residual is confirmed and is partly not the layout's fault.** `nnz >= 500` keeps 99.9% of chunks — but
it also *matches* 94.5% of rows, so most of that is a genuinely unselective predicate rather than a pruning failure.
A filter that wants almost every row cannot be helped by skipping chunks.

#### The policy this settles: prune adaptively, on a threshold the reader can evaluate for free

The index sidecar is read first and is cheap — 0.012 MB and 16 requests at 5M rows, 0.15 MB at 20M — and reading it
tells the reader **how many chunks survive before it fetches any column data**. That makes the choice measurable at
query time rather than fixed at write time.

The break-even follows from the sweep's own counters. A pruned read costs about 3 requests per surviving chunk
(2,447 requests for 830 chunks at 20M; 626 for 207 at 5M), while an unpruned read costs about one request per
shard, and a shard holds 8 chunks. Pruning therefore wins on requests when survivors fall below roughly
**(1/8) / 3 ≈ 4-5% of chunks**, which the measured configurations put at 33 of 611 chunks at 5M and 111 of 2,442 at
20M. Below that line pruning wins on both metrics at once; above it, it trades bytes for requests and 10.1's ranking
says to decline the trade.

Applied to the real predicates above, at human scale the line sits at about 889 of 19,408 chunks:

| Predicate | Chunks touched | Prune? |
|---|---:|---|
| `dataset_id ==` one dataset | 506 | **yes** — and this is the scoping clause most queries carry |
| `cell_type ==` median rank | 58 | **yes**, by two orders of magnitude |
| `cell_type ==` rarest | 1 | **yes** |
| `cell_type ==` most common | 2,411 | no — read whole shards |
| `tissue_general ==` brain | 6,361 | no |
| `nnz >= 500` | 19,396 | no |

So the index earns its keep on the selective end, which is where 9.3 located the read-path problem, and the reader
declines it on the broad end where a whole-column read is cheaper. **3.1 adopts this as the index scope rule.** What
remains unmeasured is the threshold's exact position on S3 rather than a local filesystem, since it balances a
request count against a byte count and only their relative latency decides the crossover (10.1).

**Sharding buys the object-count property outright.** At 2048-row chunks the unsharded store is **20,642 objects**
against **726** sharded, a 28x reduction, with bytes unchanged (0.11 vs 0.12 MB). Against the SOMA baseline of
325,695 objects that removes object count as a concern for a sharded layout, while showing how an unsharded one
would blow past the baseline.

**8192-row chunks are chosen on requests, not bytes.** Against 2048 under the same order and pruning, 8192 issues
1,005 requests to 2048's 1,850 on the two-column predicate and 4,664 to 7,704 on WMG, roughly 1.7x fewer, while
2048 reads 25--35% fewer bytes (3.15 against 4.19 MB; 6.54 against 8.88). 10.1's ranking decides it. Both directions
hold at 20M rows — 3,951 requests against 7,303 on two columns, and 12.57 MB against 16.75 — so the choice is not an
artefact of the smaller axis. Do not go finer than 8192 without re-measuring.

**Does it meet 9.3's target?** Partly. A single-column predicate costs 0.53 MB and two columns 4.19 MB, a **7.9x
cliff**, where TileDB's on the same `nnz`-threshold shape is 697x. The two axes differ in scale, so treat that as
an order-of-magnitude statement rather than a like-for-like ratio: the cliff is roughly 90-fold smaller, and it is
not gone. The residual is `nnz >= 500` itself — min/max statistics cannot exclude a chunk when the threshold falls
inside almost every chunk's range, so the `nnz` column is read for every surviving candidate. That is a property of
the *predicate shape*, not the layout: equality on a dictionary column prunes to almost nothing, a threshold on a
high-cardinality numeric column prunes to almost everything, which is why 3.2's diagram cannot show the `nnz`
clause being pruned. Materialising the threshold as a stored boolean would reduce that clause to a dictionary
predicate, and is rejected because the corpus should not accrete a column per consumer.

Bounding these claims: the axis is synthetic above 64,798 rows and 5M is 1/32nd of the human experiment. Reads are
local, so request counts are real and their S3 latency consequence is not measured and cannot be from here (10.1).
Only `obs` is swept — `X` is 10.3 and 10.4, and 9.4 established `X` has no gene axis to sweep. The pruning
correctness check, pruned and unpruned reads returning identical rows, passed in all 32 configurations, which is
the precondition for any of these numbers meaning anything.

### 10.3 `X` encoding: banded CSR recovers the gene axis

Four candidate encodings of the same matrix — 64,798 cells x 61,497 genes, 51,960,590 nonzeros, 1.30% dense — each
written from the rung 25 store, with every encoding required to return identical values for every read.

| Encoding | Store | Objects | 1 gene | 100 contiguous genes | all genes, contiguous rows |
|---|---:|---:|---:|---:|---:|
| `bands` (16 gene bands, CSR each) | **92.7 MB** | 217 | **0.51 MB** | 0.31 MB | **16.62 MB** |
| `csr` (AnnData's default) | 98.7 MB | 13 | 98.71 MB | 16.68 MB | 16.68 MB |
| `csc` | 115.1 MB | 13 | 4.65 MB | 4.65 MB | 115.13 MB |
| `dense` (2-D, 2048x2048 chunks) | 136.5 MB | 26 | 1.04 MB | **0.20 MB** | 22.02 MB |

**CSR reads the entire matrix to return one gene** — 98.71 MB of a 98.7 MB store for 1 of 61,497 genes. This is
9.4's finding at the store boundary, and it is total rather than partial. Banding costs 0.51 MB for the same query,
a **194x** reduction.

**Banding is the best all-rounder and has the smallest store.** Its row-selective read is within 0.4% of CSR's
(16.62 against 16.68 MB) and its contiguous gene-block read is 54x better. Splitting genes into 16 bands narrows
each band's column indices from 0--61,496 to 0--3,843 and `indices` compresses better as a result, which is why 217
objects still cost fewer bytes than CSR's 13. Its price is request count: 148--264 against CSR's 18--20.

**A dense 2-D `X` costs 1.38x the sparse store, not the 15.9 GB the arithmetic suggests.** At 1.30% density the
zeros compress to almost nothing, so the objection that has kept dense off the table is wrong by two orders of
magnitude. Dense buys true two-dimensional chunking — at 2048x2048, exactly the TileDB tile geometry of 9.2 — for
38% more storage, and it wins outright on contiguous gene blocks. It is rejected only because banding matches it
more cheaply on the queries that exist today (spec §3.3).

**CSC is the mirror-image failure and is dropped.** It reads the whole array for any row-selective query (115.13
MB) — the exact pathology CSR has on the gene axis, moved to the other axis — and is never the best choice except
for a query no consumer issues.

**Gene-axis locality only pays for *contiguous* gene selections, which no current consumer asks for.** A contiguous
100-gene block costs 0.31 MB banded and 0.20 MB dense; 100 *scattered* genes cost 16.62 and 21.59, some 50x and
100x worse, because 100 random genes hit all 16 bands and most column chunks. A real gene query is a marker panel
scattered through `feature_id` order, so recovering the benefit needs the genes *reordered* so co-queried ones are
adjacent — a second clustering decision, on the other axis, with no obvious key (spec §3.3). Without it, banding's
advantage applies to single-gene lookups and little else.

Bounding these claims: one matrix, one layer, 64,798 cells — a rung 25 store, not the corpus. Local filesystem.
`raw` only; the `normalized` layer would roughly double every store figure. Band count fixed at 16, not swept.

### 10.4 Row order: assignment order is kept

Row order is a layout choice the current corpus does not exercise, and clustering by a commonly-filtered attribute
is the obvious candidate. Measured on `X` in the banded encoding, same matrix, two real predicates, two row orders,
against the theoretical floor of store size times the fraction of nonzeros the query needs:

| Predicate | Rows | Floor | Assignment order | Clustered by cell type |
|---|---:|---:|---:|---:|
| `cell_type_ontology_term_id == 'CL:0002627'` (17.5% of nonzeros) | 11,470 | 16.26 MB | 19.55 MB (1.20x), 250 reqs | **18.42 MB** (1.13x), 206 reqs |
| `dataset_id == 'd3566d6a…'` (3.7% of nonzeros) | 15,502 | 3.42 MB | **5.92 MB** (1.73x), 152 reqs | 18.34 MB (5.4x), 314 reqs |

**Clustering gains 6% on the query it is built for and costs 210% on the other.** That is not a trade worth making,
and the reason is the useful part: **real predicates already have row locality under assignment order, because
datasets are biologically homogeneous.** A cell type concentrates in the few datasets that assayed it, so its rows
are not scattered through the corpus even though nothing sorted them. Assignment order is within **1.2x--1.7x of
optimal** for both predicates with no clustering at all, which leaves clustering very little to win and plenty to
lose.

**The `obs`-side gain from clustering is real, and larger than it first looked.** Measured on the 20M-row axis
(10.2), clustering cuts a single-column predicate from 2.10 MB to 0.11 MB and WMG from 35.50 MB to 1.87 MB, saving
2.0 MB and 33.6 MB — which scales to roughly 16 MB and 270 MB per query at the human experiment's 162M rows. It cuts
requests hard too, 2,447 to 84 on one column, because clustering is what makes the surviving chunks adjacent.

So the rejection does not rest on the `obs` term being small. It rests on the `X` term being larger, and on which
way each scales. **A clustered corpus makes a `dataset_id`-shaped selection scattered through the whole axis**, so
its `X` read touches a corpus-proportional share rather than one contiguous block: at 64,798 cells that took 5.92 MB
to 18.34 MB, essentially the whole store. The `obs` saving and the `X` penalty therefore both grow with the corpus,
and `X` is three orders of magnitude the size of one `obs` column (10.5). A few hundred megabytes saved on `obs`
does not pay for a corpus-proportional share of a terabyte-scale `X`.

**That argument is not measured at corpus scale**, and it is the weakest link in this decision. The `X` penalty is
one matrix at 64,798 cells; the extrapolation is from the mechanism, not from a number.

**Assignment order is kept.** It is dataset-contiguous, already near-optimal for both predicate shapes measured,
and free. The effort belongs on the `obs` index instead, which attacks the predicate-evaluation cost 9.3 identifies
as the dominant one and still prunes under assignment order, at 2.3--2.5x on bytes (10.2).

Bounding these claims: one matrix, 64,798 cells, one clustering key, two predicates, `raw` only. The floor
calculation assumes compression is uniform across the store, which it is not exactly. And these are the two
predicates the manifests make available at this rung; a query whose rows genuinely are scattered — a rare cell type
present in hundreds of datasets — would behave differently, and none was tested. That is the one case that could
reopen this decision (spec §3.3).

### 10.5 One store, uniform `obs` chunks, dataset-aligned `X` chunks

Two related questions: whether to split the corpus into per-dataset stores, and whether to place chunk boundaries
at dataset starts within one store. Both are measured. Per-dataset stores lose; alignment loses on `obs` and
**wins on `X`**, which is where the byte saving that motivated it was coming from all along.

**Per-dataset stores.** One store holding all 24 rung-25 datasets against 24 per-dataset stores plus a shared index
of per-dataset summaries, so a predicate can rule a store out without opening it. Both were required to select the
same cells for every query.

| Query | Stores touched | Monolithic | Per-dataset |
|---|---:|---:|---:|
| `dataset_id ==` one dataset | 1 of 24 | 5.93 MB, 167 reqs | **3.22 MB, 132 reqs** |
| `cell_type ==` | 2 of 24 | 19.56 MB, 264 reqs | **15.64 MB**, 276 reqs |
| WMG | 3 of 24 | 17.39 MB, 335 reqs | **11.58 MB**, 400 reqs |
| all datasets (`nnz >= 0`) | 24 of 24 | 92.79 MB, **275 reqs** | 93.92 MB, 3,021 reqs |
| Store total | | **93.6 MB, 268 objects** | 108.2 MB, 3,449 objects |

**Monolithic wins the moment a query fans out.** At full fan-out per-dataset issues **3,021 requests against 275**,
an 11x penalty for the same bytes, which is **114 extra requests per store opened**; extrapolated to 1,845 datasets
a corpus-wide query would add roughly **211,000 requests**, 10.1's failure mode at a scale that dwarfs the 33,054
that made it a finding. Object count scales the same way: 144 objects per dataset store implies about 265,000 at
1,845 datasets, close to the SOMA baseline, where a monolithic store grows with data volume and sits at 268 objects
for 93.6 MB.

**And the fan-out is measured, not assumed.** 11.2 records the fan-out of the client's own documented filters over
the 1,086 human datasets — the baseline is `get_anndata`, not any one downstream pipeline. At 114 requests per
store, `tissue_general == 'vasculature'` touches 10 datasets for ~1,100 requests against monolithic's ~275;
`is_primary_data == True and tissue_general == 'lung'` touches 45 (~5,100); `tissue_general == 'lung'` touches 92
(~10,500); `tissue_general in ['brain', 'lung']` touches 447 (~51,000). The measured crossover is low — per-dataset
is cheaper at one store, level at two, behind at three — and every documented filter touches at least ten, so
per-dataset loses on documented usage by 4x to 185x on requests.

**Dataset-aligned chunk boundaries within one store.** The per-dataset byte advantage is boundary alignment: in a
monolithic store a dataset's rows share their first and last chunks with its neighbours', so those rows are fetched
and discarded. Alignment promises that saving without the per-store request cost. Measured on the same 64,798-row
`obs`:

| Metric | `aligned_chunks` (exact, unsharded) | `aligned_shards` (rounded, sharded) | `uniform` (8192 / 65536) |
|---|---:|---:|---:|
| Rows scanned, `dataset_id ==` | **15,502** — exactly the dataset | 21,504 | 24,576 |
| Rows scanned, WMG | **3,752** | 34,816 | 40,960 |
| kB read, `dataset_id ==` | 16.7 | 27.1 | **5.5** |
| kB read, client 2-column | 17.4 | 33.2 | **6.4** |
| kB read, WMG | 46.1 | 146.6 | 118.0 |
| Objects | 229 | 232 | **50** |
| Index size | 23.5 kB | 39.3 kB | **9.6 kB** |

**Alignment prunes an order of magnitude better and still reads 3x more bytes.** It does exactly what it was
designed to on rows scanned, but dataset-sized chunks are small — median 1,324 rows against uniform's 8,192 — and
small chunks compress badly. This is 10.2's chunk-size result restated: on the `obs` axis compression efficiency
dominates, and a layout that shrinks chunks to gain precision loses more to the compressor than it saves in rows.
`aligned_shards` is the useful negative result — rounding each dataset up to a multiple of 512 rows keeps sharding
but gives up exactness, and it scans nearly as much as uniform while paying alignment's object and index costs.

A second reason not to ship it: variable-width chunks exist as Zarr v3's *rectilinear chunk grid*, but in zarr 3.2.1
they are **experimental and disabled by default** (`zarr.config.set({'array.rectilinear_chunks': True})`) and
**cannot be combined with sharding**. Zarr directs you instead to rectilinear *shards* with uniform chunks inside,
each shard a whole multiple of its inner chunk; since real dataset widths at this rung run 146 to 15,502 rows,
exact alignment and sharding are mutually exclusive. A recommendation resting on an experimental feature and
forfeiting either sharding or exactness is not one this design makes (spec §5).

**Where the 20--46% byte saving actually lived: `X`.** The store-organization figures are megabytes and the `obs`
alignment figures kilobytes, because those queries read `obs` **and** `X` and `X` dominates by three orders of
magnitude. The saving was on the `X` side — a dataset's nonzero ranges in a banded CSR array stop sharing chunks
with its neighbours' — and almost none of it was `obs`. So the decision splits by axis, and the `X` half is
measured below.

Bounding the `obs` half: 24 datasets, one rung, one organism. The fan-out extrapolation assumes the per-store cost
is constant, which it should be since it is metadata. The per-dataset store duplicates every category dictionary,
which is most of its 16% size overhead and would grow with dataset count. Dataset widths here (146 to 15,502 rows)
are far smaller than the corpus median, and alignment's compression penalty shrinks as datasets grow, so a
corpus-scale test could land differently — though it would have to overcome a 3x gap. Local filesystem throughout.

#### `X`: aligning nonzero chunk edges to datasets, by padding

Exact alignment needs Zarr v3's rectilinear chunk grid, which is experimental in zarr 3.2.1 and **cannot be
combined with sharding**, so it is not a candidate whatever it reads. The buildable form keeps uniform chunks and
**pads** each dataset's nonzero range up to a chunk multiple, so no chunk holds two datasets' nonzeros while
sharding survives. Padding entries carry `indices = -1`, which is outside every band's gene range, so the *same*
reader reads both layouts and the comparison is layout-only; `test_x_aligned.py` asserts value-identical results and
mutates the sentinel to prove the assertion bites. Same matrix as 10.3 — 64,798 cells, 51,960,590 nonzeros, banded
CSR, 16 bands, 125,000-element chunks per band. "Floor" is the store size times the fraction of nonzeros the query
needs, as in 10.4.

| Query | Uniform | Padded | Change | vs floor |
|---|---:|---:|---:|---:|
| `dataset_id ==` one dataset | 5.92 MB, 152 reqs | **3.86 MB, 142 reqs** | **−34.8%** | 1.73x → **1.13x** |
| `cell_type ==` cross-dataset | 19.55 MB, 250 reqs | **16.19 MB**, 282 reqs | **−17.2%** | 1.20x → **1.00x** |
| WMG | 17.27 MB, 266 reqs | **12.17 MB**, 268 reqs | **−29.6%** | 2.23x → 1.57x |
| Store | 92.7 MB, 217 objects | **92.6 MB**, 239 objects | −0.1% | |

**Padding recovers the saving and costs nothing to store.** 17--35% fewer bytes on every query, and the cell-type
read lands exactly on its floor. The storage objection — the reason this was expected to be a trade — does not
survive contact: 19.16 million padding elements, 37% as many as there are real nonzeros, changed the store by
**−0.1%**, because a run of identical zeros and −1s compresses to almost nothing and aligned chunks hold one
dataset's values rather than two datasets' mixed together. Object count rises 10%, and requests are a wash: −10 on
the dataset query, +32 on the cell-type one, +2 on WMG.

**It also beats the obvious alternative, which is simply using a smaller chunk.** Shrinking the chunk reduces
edge spillover too, and the same run at 25,000 elements per band shows what that costs:

| Layout | `dataset_id` read | Requests | Store |
|---|---:|---:|---:|
| uniform, 125,000/band | 5.92 MB | 152 | 92.7 MB |
| uniform, 25,000/band | 4.31 MB | 272 | 98.4 MB |
| **padded, 125,000/band** | **3.86 MB** | **142** | **92.6 MB** |

Padding at the coarse chunk dominates the fine uniform chunk on all three metrics at once. That is the same
compression effect 10.2 found on `obs`, in the other direction: fine chunks buy precision and lose it again to the
compressor, where alignment buys the precision without shrinking anything. **3.1 therefore pads `X` nonzero ranges
to dataset boundaries at the standing 125,000-element chunk.**

Bounding these claims: one rung, one organism, `raw` only, 24 datasets, local filesystem. The magnitude is
scale-sensitive in a way the direction is not — the saving is edge spillover, which is roughly the chunk size over a
dataset's nonzeros per band, and at this rung that ratio is about 0.9 because a dataset barely fills one chunk.
Corpus datasets are much larger, so the percentage should fall; the 25,000-element run shows exactly that
dependency, with the saving dropping to 4--8% when the ratio drops fivefold. **Do not quote 17--35% as a
corpus-scale figure.** What does carry is the dominance argument above, which is structural, and the storage cost,
which is nil at both chunk sizes.

## 11. Validation methodology

The metrics, query corpus and pass conditions were fixed before the measurements of Sections 9 and 10 were taken,
so those results are reported against criteria that were not chosen after seeing them. One criterion did change:
10.1's finding on wall clock, and 11.4 states the revised form.

### 11.1 v1-over-TileDB is the oracle

A correct implementation of every claim in this document already exists and keeps running throughout the
transition: the v1 client reading the TileDB-SOMA corpus. Equivalence is therefore established **differentially**,
by running the same query through both stacks and comparing results rather than against expectations written by
hand. That is what makes validation meaningful at one to five datasets instead of requiring full corpus scale, and
it is why the dual-publish window of 7.1 has testing value beyond continuity: while both formats exist, every claim
remains checkable.

The oracle expires at cutover. Until then, two conditions keep it trustworthy:

- **Every recorded result names the oracle version.** `tiledbsoma` is on an active release cadence and is pinned at
  1.15.3 in the builder (11.7). A parity result measured against an unrecorded oracle version cannot be reproduced.
  The same applies to `anndata`, whose `~=0.12` pin floats across minor releases (11.7).
- **A disagreement is not automatically a v2 defect.** Where the stacks differ, the source datasets are the
  tiebreaker, since both derive from them. The oracle can be wrong: a discrepancy may be a pre-existing TileDB-path
  bug, which is a finding to report upstream rather than a v2 failure to fix.

### 11.2 Query corpus

A fixed set of queries, versioned alongside the harness, each earning its place by exercising something that could
plausibly break independently.

**The baseline is the v1 Python client, not any downstream service.** What v2 must not regress against is
`cellxgene_census.get_anndata()` — the call the API contract and the 5-year LTS commitment are made about (Sections
6, 7). Every SOMA-side figure in Sections 9 and 10 is that path: `readers.read_soma` builds an `axis_query` and
calls `to_anndata()`, which is what `get_anndata` does internally, so the oracle of 11.1 and the performance
baseline are the same code.

**The primary anchor is what the client's own documented usage does.** Filters taken from this repo's notebooks,
with their measured fan-out over the 1,086 human datasets:

| Documented client filter | Cells | Datasets touched |
|---|---:|---:|
| `tissue_general == 'vasculature'` | 431,901 | 10 (1%) |
| `is_primary_data == True and tissue_general == 'lung'` | 6,167,731 | 45 (4%) |
| `tissue_general == 'lung'` | 10,181,125 | 92 (8%) |
| `tissue_general in ['brain', 'lung']` | 58,762,981 | 447 (41%) |

These are one- and two-column predicates over corpus-wide attributes, which matters twice over. The two-column
cliff of 9.3 is not an artificial construction — `is_primary_data == True and tissue_general == 'lung'` is exactly
that shape, it is in the documentation, and it costs 9.2x — and the fan-out range that decides store organization
(10.5) is **10 to 447 datasets**, measured rather than inferred. The same measurement gives the per-column costs 9.3
rests on: `dataset_id`, `tissue_general` and `is_primary_data` together cost **4 MB to scan across 158,982,719
rows**, 0.025 bytes per row, because all three correlate with row order and compress to almost nothing. Against
`nnz` at 1.6 bytes/row and `cell_type` at 0.33, predicate cost varies about 65x with *which* columns are named, and
the expensive clauses are the numeric ones.

WMG sits in the corpus as what it is: a real filter shape from a real consumer, useful because it is the widest
fan-out anyone has asked for and the only 4-column predicate available. It is not the baseline, and its own cost
today is unknown — no WMG wall-clock baseline exists in recorded form, and 10.1 shows one taken from outside the
region would have been unusable anyway.

| Query | What it exercises |
|---|---|
| The documented client filters above | The baseline the no-regression threshold is set against |
| WMG's production filter, verbatim below | A real 4-column predicate at maximum fan-out |
| Single-gene lookup across all cells | Worst-case sparse column access |
| Broad tissue slice | Large contiguous read |
| Cell-type slice | Selective read with a dictionary-encoded predicate |
| Identity-coordinate access | `soma_joinid` on v1, its replacement on v2 (spec §4) |
| Presence-matrix read | The measured-vs-zero distinction (spec §2) |
| Spatial read: scene, point cloud, image | The four object kinds with no non-spatial counterpart (spec §2) |
| Multi-valued `disease` filter | Membership rather than string equality (spec §2) |
| Symbol-collision `feature_name` lookup | Non-unique gene symbols (spec §2) |
| `raw` and `normalized` on non-spatial; `raw` only on spatial | Layer presence varying by experiment (spec §2) |

**The WMG filter, verbatim**, from `backend/wmg/pipeline/constants.py` in `single-cell-data-portal`, fully
interpolated. Human and mouse use an identical string:

```
is_primary_data == True and nnz >= 500 and cell_type_ontology_term_id != 'unknown'
and tissue_general_ontology_term_id != 'UBERON:0001017'
and tissue_general_ontology_term_id != 'UBERON:0001007'
and tissue_general_ontology_term_id != 'UBERON:0002405'
and tissue_general_ontology_term_id != 'UBERON:0000990'
and tissue_general_ontology_term_id != 'UBERON:0001004'
and tissue_general_ontology_term_id != 'UBERON:0001434'
```

The field is `cell_type_ontology_term_id`, not `cell_type`, and the threshold is `nnz >= 500`. The read is
`census_version="latest"`, two organisms sequentially, measurement `RNA`, **no `var` filter at all**, and
`X["raw"]` consumed blockwise on axis 0 at `row_stride = 50_000` as CSR. Two inefficiencies in it affect fairness:
the pipeline materialises the full filtered `obs` to pandas **twice**, in `expression_summary.py` and again in
`cell_counts.py`, and passes no `column_names` projection, so every `obs` column is read. A comparison must either
reproduce both or state plainly that it did not.

Two rules on coverage. **The corpus iterates experiment specs rather than hardcoding human and mouse**, because
there are seven experiments (spec §2) and a result covering two of them is not a result about the Census. And **the last
four rows exist because they fail silently**: a filter that treats a multi-valued field as a string, or resolves a
gene by a colliding symbol, returns a plausible subset and raises nothing.

**Matched pairs, for the benchmark rather than for parity.** Amplification is query-dependent, so ranking layouts
needs pairs that vary one axis and hold the rest fixed, each sized to make any difference attributable to locality
rather than volume:

| Pair | Varies | Status |
|---|---|---|
| `dataset_id == X` vs `cell_type_ontology_term_id == Y` | row locality at similar result size | Needs a third variant: its halves differ in predicate column count as well as locality (9.2) |
| all genes vs a ~100-gene subset, same cells | column locality | Cannot discriminate as written — sparse `X` has no gene axis (9.4). Becomes a measurement only when the *encoding* varies, which is 10.3 |
| 0.1% vs ~50% selective, same experiment | selectivity | Live |

### 11.3 Parity checker

Runs a query through both stacks and asserts:

1. **Identical cell selection**, compared by source `dataset_id` plus the source H5AD's own row identity, not by
   either stack's internal join key — comparing by `soma_joinid` would assume the mapping under test (spec §4).
2. **Identical gene selection**, compared by `feature_id` (spec §2).
3. **`X` values equal within tolerance**, for every layer the experiment carries. `raw` holds integer counts and
   must match exactly; `normalized` is floating point and is compared within a stated tolerance.
4. **`obs` and `var` columns equal**, including dtype and categorical level sets, since a dictionary-encoded column
   that loses its levels still compares equal value-by-value (9.5).

Parity is a precondition, not a tradeoff: a faster stack that fails parity fails the gate.

### 11.4 Benchmark rig

Reported per query, per dataset-count, and per layout candidate. **The metrics are ranked, not a flat list**,
because wall clock from outside `us-west-2` varies by more than an order of magnitude on an identical query and
cannot carry a comparison (10.1).

**Primary: deterministic counters.** Reproducible across machines and networks, so a difference between two layouts
is a real difference.

| Metric | Why | How measured |
|---|---|---|
| Request count | Dominates cold reads (10.1), and the metric object-per-chunk layouts regress | Zarr: wrap the store and count `get`/`get_partial_values`. TileDB: `Context.VFS.read_ops_num` |
| Bytes read | Distinguishes a slow stack from an over-fetching one | Zarr: sum returned buffer lengths. TileDB: `Context.VFS.read_byte_num` |
| Corpus bytes and object count | Dual-publish storage cost, against the 2.7 baseline of 1.4 TiB / 325,695 objects | Store enumeration |

On the TileDB side, count **LIST and HEAD operations too**: `Context.VFS.ls_num`, `is_object_num` and
`file_size_num` are real S3 round-trips that `read_ops_num` excludes. Counting reads alone flatters TileDB.

**Secondary: wall clock, and only in-region.** Filter time and `X` slice throughput, split cold from warm, on an
EC2 instance in `us-west-2` rather than a laptop. A laptop number is reported only as a sanity check on ordering.

Runs are reported at **1, 5, 25, and 100 datasets** so the output is a scaling curve rather than a single number,
and any extrapolation to 1,845 datasets is stated with the curve it came from. The ladder takes the smallest
datasets available, so it scales dataset *count* rather than bytes; the corpus median dataset is far larger than the
ladder mean, and a large-dataset case is needed before the extrapolation is trusted. Selectivity is a dimension
rather than an accident: each query records what fraction of the experiment it selects, and the corpus deliberately
spans both ends.

**The threshold's baseline is `cellxgene_census.get_anndata()` on the published TileDB corpus**, measured in
requests and bytes rather than seconds, against the documented client filters of 11.2. 9.3 gives it a concrete
form: the current client answers a one-column scoped read in **0.4 MB and 9 requests** and a documented two-column
read in **3.4 MB**. Those are the figures v2 is held to. The 258 MB an `nnz` clause costs is the worst case rather
than the target — worth fixing, since WMG issues one, but a threshold set there would clear a bar users never
approach.

**The tolerance band: no more than 2x the v1 figure on either metric, per query, and no regression at all on the
corpus-level totals.** Two-fold rather than parity because v2 is a different engine with different metadata
overheads, and because the counters are exact (10.1), so a 2x band is a real bound rather than noise absorption.
Corpus bytes and object count get no allowance because 10.2 measured them as strictly better.

### 11.5 Dataset subsets

Canonical manifests at 1, 5, 25, and 100 datasets, each pinned by explicit `dataset_id` list rather than by
selection criteria, so results are comparable across runs and across people. The subsets must include spatial
datasets and at least one dataset from each of the five organisms; a random sample would draw almost entirely from
human and mouse (spec §2). The current manifests have a known defect: size-ascending selection concentrates secondary
data, which disables the WMG query below rung 25 (9.1).

### 11.6 What this harness does not establish

- **Full-scale behaviour.** Every result below 1,845 datasets is an extrapolation. Section 13 addresses this and
  only a full build settles it.
- **Anything the sweep's synthetic `obs` axis implies about value distribution.** It is grown past 64,798 rows by
  repeating whole dataset blocks (10.2), which keeps `dataset_id` contiguous and cell types realistically
  distributed *within* a dataset, but fixes how widely a value spreads *across* chunks: the fraction of chunks a
  predicate must read is 0.339 at 5M rows and 0.340 at 20M. So sweep ratios hold at one distribution and no row
  count changes them, and that distribution is pessimistic — the real corpus puts the same predicate shape at 0.124
  and a median-frequency one at 0.0030 (10.2). **Read sweep costs as an upper bound.** Occupancy itself is now
  measured on the real corpus, so the gap this bullet used to describe is closed; what is not closed is where the
  4--5% pruning threshold sits on S3 rather than a local filesystem, since it trades requests against bytes and only
  their relative latency fixes the crossover (10.1).
- **Errors common to both stacks.** Differential testing cannot detect a mistake made identically on both sides,
  such as a shared misreading of the source schema.
- **Real cold-S3 latency at scale.** Small-subset cold reads approximate it; a full corpus in its published
  location does not exist until the first full build.

### 11.7 Toolchain versions, and why each result must record them



| Dependency    | Builder                | Reader (`cellxgene_census`)                   |
| ------------- | ---------------------- | --------------------------------------------- |
| `tiledbsoma`  | `==1.15.3`             | `>=1.15.3` (must not fall behind the builder) |
| `anndata`     | `~=0.12`               | unpinned                                      |
| `spatialdata` | `>=0.5.0` (test extra) | `>=0.5.0` (`spatial` extra)                   |


The builder already requires `anndata~=0.12`. The Zarr v3 / lazy-read generation this design depends on is
already a census build dependency.

Two qualifications on that, both relevant to how much reassurance it offers:

- **The lazy APIs are experimental.** `read_lazy` and `read_elem_lazy` live in `anndata.experimental` and are
  documented as subject to change. Every result must therefore record the resolved `anndata` version, the same
  discipline Section 11.1 applies to the oracle.
- **But `read_elem_lazy` is already in production use in this repo**, at `build_soma/build_soma.py:475`, where
  the builder lazy-reads spatial images before handing them to a dask store. So the dependency on experimental
  lazy reads is not a new risk this design introduces; it is one the census build already carries.

Note what `~=0.12` actually permits. Under PEP 440 the two-component compatible-release form means
`>=0.12, <1.0`, **not** `>=0.12, <0.13`; only `~=0.12.0` would restrict to the 0.12 series. So the builder
already accepts every anndata minor release below 1.0, and a resolution performed on 2026-07-29 selects
**0.13.2**.

The consequence runs opposite to the obvious reading: the builder's anndata version is not pinned in any
practical sense, it floats across minor releases. Two builds months apart can therefore be produced by
different anndata minor versions with different lazy-read behaviour, and nothing records which was used. The
version-recording discipline Section 11.1 imposes on the oracle applies to the builder for the same reason.


## 12. Identity transition proof

`TBD` until the mapping artifact exists. A real contributed embedding from `cellxgene-contrib-public`, mapped through the identity model (spec §4) and
re-joined against the new corpus, landing on the same biological cells it did under v1.

### 12.1 What the proof has to assert

The identity model (spec §4) is a design, and its claim is falsifiable: an embedding row computed against v1 must, after mapping,
attach to the same cell in v2. The gate is a differential test with the same shape as the parity checker (11.3),
and the same discipline applies — it must be possible for it to fail.

Four assertions, in the order they can go wrong:

1. **The mapping is total over the release.** Every `soma_joinid` in the v1 experiment appears exactly once in the
   4.4 mapping. A mapping that silently omits cells produces an embedding that loads and is short.
2. **The mapping is injective.** No two `soma_joinid` values map to the same `(dataset_id, observation_joinid)`.
   4.1 verifies the pair is unique in the corpus, which makes this a check on the artifact rather than the data.
3. **A mapped embedding lands on the same cells.** Take an embedding's `soma_joinid` coordinates, map them, resolve
   them in v2, and the resulting cells must equal the v1 cells by source identity. This reuses the parity checker's
   cell key directly, since it is already `(dataset_id, observation_joinid)`.
4. **An unmappable identity raises.** A `soma_joinid` absent from the mapping — a cell dropped between releases —
   must produce an error, not a silently shorter result (spec §4.3).

The paired mutations that prove the gate is live: drop one row from the mapping (1 must fail), duplicate one target
pair (2 must fail), shift the mapping by one position (3 must fail), and pass a `soma_joinid` beyond the
experiment's range (4 must raise).

### 12.2 What is available to test with

`cellxgene-contrib-public` publishes contributed embeddings with a `census_version` in their metadata, and
`get_all_available_embeddings` enumerates them, so a real embedding pinned to the 2025-11-08 release can be
selected without guessing. The mapping artifact itself does not exist yet — it is produced by the builder (spec §4.3) — so
the proof needs either a builder change or a mapping derived from the published corpus, which is the same
two-column projection and costs 1--19 MB per small experiment (spec §4.3). Choosing the smallest experiment that has a
contributed embedding keeps the whole proof laptop-scale.

## 13. Build resource envelope

`TBD` until the builder emits Zarr. Memory and runtime as dataset count grows, and the projection to full scale that sizes the build
host.

### 13.1 What is already known, from the spike's own conversions

The spike converted census slices to Zarr repeatedly, and those conversions are the same operation the builder
performs, at small scale: 565 cells and a 2.1 MB store at negligible peak memory, 5,716 cells and 18.3 MB likewise,
and 64,798 cells and 213.6 MB at roughly **1.2 GB resident**.

The conversion path holds the whole matrix in memory as CSR before writing (`convert.py`), so peak memory scales
with nonzeros rather than with the store. Extrapolating the rung-25 figure on nonzeros alone puts a 340,000-cell
rung at roughly 4 GB and the full human experiment far beyond a laptop.

Two things this establishes and one it does not. It establishes that **the builder must stream rather than
materialise**, since a whole-experiment CSR is not resident-memory-shaped, and that **the banded encoding of 10.3
makes streaming natural**, because each gene band is written independently and a band is a bounded slice of the
nonzeros. The `X` padding of 10.5 helps here rather than complicating it: since every dataset's nonzero range
begins on a chunk boundary, the builder can append one dataset at a time without ever reading back or rewriting a
chunk another dataset shares. It does not establish the envelope: these are conversions of a corpus that already
exists, not builds from source H5ADs, and the builder additionally holds harmonization state this path never
touches.

`TBD`: the actual envelope, which needs the builder emitting Zarr rather than the spike converting it.

---



## 14. Findings that need a decision elsewhere

The decisions themselves are listed in spec §10, which is the authoritative set. This section holds what was
found while measuring: defects in existing documentation, and two questions only the census team can answer.

### Defects in existing documentation

In `docs/cellxgene_census_schema.md`:

1. The obs-to-scene mapping is named `obs_spatial_presence` in the implementation and at line 952, but
  `obs_presence_matrix` at lines 998 and 1033. The latter object does not exist.
2. Line 935 states spatial experiments live in `census_obj["census_data"][organism]`; should be
  `census_spatial_sequencing`.

In `dev-docs/cellxgene_census_storage_and_release_policy.md`, five points of drift, tabulated in spec §5.
The filename and bucket are wrong, the `V#` alias requirement is unmet in production, the `stable` alias is
undocumented despite being the client default, and `do_not_delete` and `relative_uri` are undocumented. This
document is the one spec §5 amends, so reconciling it is part of that work rather than separate from it.

On the `V#` alias specifically, which of the five is least obvious how to settle: the policy states one "MUST
exist that points to the release" for every LTS release, six releases carry `flags.lts`, and no such alias
exists. Nothing consumes it either. The client resolves aliases generically, `stable` is its default, and the
docsite directs users to pin by build date. So implementing it, deleting the requirement, or downgrading it to
`MAY` all appear to cost nothing, which is what makes it cheap to settle while the document is already open.

### Questions for the census team

1. **Who approves an amendment to the storage and release policy?**
  `dev-docs/cellxgene_census_storage_and_release_policy.md` must be amended
  before a Zarr corpus can be published (spec §5). The file records "Last edited: April, 2023" and has no
   substantive commit in the repository's visible history; the only change touching it is a link fix.
   Sustained recent ownership of the governing documents in `dev-docs/` and of the Census schema sits with
   **Brian Raymor** (7 of the last 10 `dev-docs/` commits, plus the 2.4.0 schema work through Nov 2025), which
   is the best available signal but not a confirmation.
2. **Is** `soma_joinid` **intended to be stable across releases?** It effectively is today, which is why users join
  on it, but the guarantee does not appear to be documented. This materially changes the identity model (spec §4), and is decision 6 there.
