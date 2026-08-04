# Getting the Census onto AnnData-Zarr: a plan

**Status:** draft, for discussion. **Last edited:** 2026-08-04.

How the design in [census_zarr_design.md](./census_zarr_design.md) gets built. The design says what the corpus
should look like; this says in what order it gets made, and what has to be true before each piece can start.

No dates or effort estimates here. The phases are ordered by dependency, not by calendar, and several can run
alongside each other where nothing forces them apart.

---

## The shape of the work

Phase 0 is done. Three things then gate everything else, and they are independent of each other, so they should
start together: approvals, because nothing lands upstream without them; the validation harness, because it is
how every later phase is checked and it stops being possible once the old stack retires; and spatial
representation, because it is the one layout question still open and the writer cannot emit spatial experiments
without an answer.

After those, the chain is: writer, then client, then a full-scale build, then publish, then migrate, then
retire. R parity runs alongside from the point the tooling allows it.


| #   | Phase                    | Gated on                           |
| --- | ------------------------ | ---------------------------------- |
| 0   | The design               | Done                               |
| 1   | Approvals and amendments | Decisions in spec section 10              |
| 2   | The validation harness   | Nothing                            |
| 3   | Spatial representation   | Nothing                            |
| 4   | The builder writes Zarr  | 1, and 3 for spatial               |
| 5   | The v2 Python client     | 4                                  |
| 6   | The identity proof       | 4, 5                               |
| 7   | The full-scale build     | 4, and 3 for spatial               |
| 8   | Publish                  | 1, 6, 7                            |
| 9   | Migrate the ecosystem    | 8                                  |
| 10  | R parity                 | Bioconductor tooling, 8            |
| 11  | Cutover                  | Everything, and the support window |


---



## Phase 0: Design spec

**Delivers** the specification, and the measurements behind it.

**Done when t**he layout is decided and measured, and nothing later revisits it.


| Settled                                                           | Where                                                                            |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| The read path's cost problem, measured against the current format | [Evidence section 9](./census_zarr_design_evidence.md#9-read-path-feasibility-evidence) |
| The storage layout, chosen against measured alternatives          | [Evidence section 10](./census_zarr_design_evidence.md#10-layout-comparison)            |
| How cells and genes are identified                                | [Spec section 4](./census_zarr_design.md#4-how-cells-and-genes-are-identified)          |
| What the v2 client surface looks like                             | [Spec section 6](./census_zarr_design.md#6-the-python-api)                              |


**Watch for** what is not covered. Spatial representation is Phase 3. Every corpus-scale figure is an
extrapolation until Phase 7 replaces it with a measurement. And none of this exists in the build yet: no writer
emits this layout, no client reads it, and no corpus has been built at full scale.

---



## Phase 1: Approvals and amendments

**Delivers** Census schema 3.0.0, an amended storage and release policy, and a docsite that no longer defines a
release as a TileDB-SOMA collection.

**Needs first** answers to the decisions in [spec section 10](./census_zarr_design.md#10-decisions-requested).
Decisions 1 and 2 are the blocking pair. Decision 5, on whether `soma_joinid` was always intended to be stable,
should be answered early because a "yes" would send [spec section 4](./census_zarr_design.md#4-how-cells-and-genes-are-identified)
back for rework rather than adjusting it.

**Done when** all three documents are merged upstream, and the policy amendment carries a check that fails a
release which does not satisfy it.

**Watch for** the approver of the release policy being unidentified, and the five-year availability commitment
living on the docsite rather than in the policy, so amending the policy alone does not touch it.

---



## Phase 2: The validation harness

**Delivers** a differential checker that runs a query through both stacks and compares results, a fixed query
corpus, and a benchmark rig reporting bytes and requests rather than elapsed time.

**Needs first** nothing. It can start immediately and should, because everything after it is checked with it.

**Done when** the harness runs both stacks over a handful of datasets and agrees on cells, genes, values and
metadata columns, and deliberately corrupting a result makes it fail. A check that cannot fail is not a check.

**Watch for** the reference expiring. The old stack is the oracle, so anything wanting differential proof has
to be proven before the TileDB corpus retires. Method is in
[evidence section 11](./census_zarr_design_evidence.md#11-validation-methodology).

---



## Phase 3: Spatial representation

**Delivers** a decision on how scenes, point clouds, images and the cell-to-scene mapping are stored, and the
accessor that reads them.

**Needs first** nothing, but it blocks the writer for two of the seven experiments.

**Done when** the mapping to `spatialdata` is specified in the schema, or spatial is explicitly deferred and
the corpus ships non-spatial first.

**Watch for** this being the only layout question still open ([spec section 3.3](./census_zarr_design.md#33-open-questions)),
and spatial being 2 of 7 experiments, so deferring it is a real option rather than a failure.

---



## Phase 4: The builder writes Zarr

**Delivers** the writer layer: banded CSR with padded dataset-aligned chunks, chunked and sharded obs, the
per-chunk index, and the identity mapping table.

**Needs first** Phase 1, since the schema has to define where the index lives before a build can emit one. Also
Phase 3 for spatial experiments, though non-spatial can proceed without it.

**Done when** a differential build at increasing dataset counts produces output equal to a TileDB build from
the same inputs, cell for cell and value for value.

**Watch for** memory. The spike held whole matrices in memory, which does not scale to an organism, so the
writer must stream. The layout helps: each band is written independently, and because every dataset starts on
a chunk boundary, datasets can be appended one at a time without rewriting a neighbour's chunk.

---



## Phase 5: The v2 Python client

**Delivers** the surface in [spec section 6](./census_zarr_design.md#6-the-python-api): opening a corpus, the
accessors, filter translation, selection by the new identity, and lazy access.

**Needs first** Phase 4, or at least enough of it to produce a store to read.

**Done when** every query in the harness corpus returns identical results through the old stack and the new
one.

**Watch for** the filter translator treating `disease` and `disease_ontology_term_id` as multi-valued, which
fails silently rather than loudly if missed. The `experimental` namespace is not scoped in the design and needs
its own decision, particularly `get_embedding`, which is keyed on the identifier being retired.

---



## Phase 6: The identity proof

**Delivers** a real contributed embedding, mapped through the published mapping and re-joined against the new
corpus, landing on the same biological cells.

**Needs first** the mapping artifact from Phase 4 and the client from Phase 5.

**Done when** the four assertions in
[evidence section 12](./census_zarr_design_evidence.md#12-identity-transition-proof) hold, each paired with a
corruption that makes it fail.

**Watch for** this being a release gate rather than a nice-to-have. It must be demonstrated against a real
embedding, not a synthetic one.

---



## Phase 7: The full-scale build

**Delivers** the first complete Zarr corpus, and the build's real memory and runtime envelope.

**Needs first** Phase 4, and Phase 3 if spatial is included.

**Done when** a full-corpus differential against a TileDB build passes, and the corpus-scale figures replace
the extrapolations the design currently carries.

**Watch for** this being the phase that tests every claim marked as an extrapolation in
[spec section 8](./census_zarr_design.md#8-risks). Expect to update the design with real numbers rather than to
confirm the estimates exactly.

---



## Phase 8: Publish

**Delivers** a release carrying the Zarr corpus, its locator in the release index, and the identity mapping.

**Needs first** Phases 1, 6 and 7.

**Done when** a published release is readable by the v2 client from S3, and the mapping resolves for a real
embedding.

**Watch for** this being the point the move becomes mandatory. There are no further TileDB releases after it, so
the timeline has to be announced ahead of the release rather than alongside it. The identity mapping is derived
from the last TileDB release.

---



## Phase 9: Migrate the ecosystem

**Delivers** notebooks, docsite and tutorials running against the new corpus, and a migration path published
for external users.

**Needs first** Phase 8.

**Done when** the notebooks and vignettes execute green against the new corpus, and canary users outside the
team have migrated successfully.

---



## Phase 10: R parity

**Delivers** an R client that reads the published corpus.

**Needs first** sharding support reaching a Bioconductor release, expected around October 2026, and Phase 8
for something to read.

**Done when** R and Python agree on the harness query corpus.

**Watch for** the cheap early check being worth doing now rather than here: confirming that `anndataR` over a
development `Rarr` really does read a sharded store, since neither project appears to test that combination.
Until R can read Zarr, R users see no new data at all, since every release after Phase 8 is Zarr.

---



## Phase 11: Cutover

**Delivers** retirement of the TileDB corpus and the v1 client.

**Needs first** everything above, and the support window.

**Done when** the support matrix is retired per the availability commitment.

**Watch for** this being policy-gated rather than readiness-gated, and constrained by R: retiring TileDB before
R can read Zarr removes access for those users entirely.

---



## Running alongside

These are not phases and do not gate anything, but they will not happen unless someone owns them.

- **Keeping the design current.** Phase 7 replaces extrapolations with measurements, and Phase 1 may change
section 4. Both should land back in the design rather than only in someone's head.
- **Scoping the** `experimental` **namespace**, which is SOMA-coupled and outside the current design.
- **Upstream documentation defects** found while measuring, listed in
[evidence section 14](./census_zarr_design_evidence.md#14-findings-that-need-a-decision-elsewhere). Cheap to fix and
they build goodwill with the team whose approval Phase 1 needs.

---



## Open questions for this plan

- Can Phases 4 and 5 overlap, with the client built against a partial store, or does the writer need to be
finished first?
- Is spatial in the first published corpus, or deferred so non-spatial can ship sooner?
- Does the full-scale build need infrastructure that has to be requested before Phase 7 can start?
- Who owns each phase? Nothing here is assigned.

