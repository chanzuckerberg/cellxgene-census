# README

This is a tool to build the SOMA instantiation of the Cell Census schema, as specified in this doc:

https://docs.google.com/document/d/1GKndzCk9q_1SdYOq3BeCxWgp-o2NSQkEmSBaBPKnNI8/

CAVEATS (READ THIS):

1. The code is written to the still-rapidly-evolving and **pre-release** Python SOMA API, _and will be subject to change_ as the SOMA API and `tiledbsoma` evolve and stabilize.
2. The schema implemented by this code is still evolving and subject to change.
3. The `cell_census_builder` package requires Python 3.9 or later.

## Usage

TL;DR:

- given a set of H5AD files, which comply with cellxgene 3.0 schema,
- create several SOMAExperiment aggregations representing mouse & human slices of the entire collection, and
- embed experiments into a single SOMACollection, along with other metadata about the aggregation/census

The build process:

- Step 1: Retrieve all source H5AD files, storing locally (parallelized, I/O-bound)
- Step 2: Create root collection and child objects (fast).
- Step 3: Write the axis dataframes for each experiment, filtering the datasets and cells to include (serialized iteration of dataset H5ADs).
- Step 4: Write the X layers for each experiment (parallelized iteration of filtered dataset H5ADs).
- Step 5: Write datasets manifest and summary info.
- (Optional) Consolidate TileDB data 
- (Optional) Validate the entire Cell Census, re-reading from storage.

Modes of operation:
a) (default) creating the entire "cell census" using all files currently in the CELLxGENE repository.
b) creating a smaller "cell census" from a user-provided list of files (a "manifest")

### Mode (a) - creating the full cell census from the entire CELLxGENE (public) corpus:

- On a large-memory machine with _ample_ free (local) disk (eg, 3/4 TB or more) and swap (1 TB or more)
- To create a cell census at `<census_path>`, execute:
  > $ python -m cell_census_builder -mp --max-workers 12 <census_path> build
- Tips: 
    - `-v` to view info-level logging during run, or `-v -v` for debug-level logging
    - `--test-first-n <#>` to test build on a subset of datasets
    - `--build-tag $(date +'%Y%m%d_%H%M%S')` to produce non-conflicting census build directories during testing

If you run out of memory, reduce `--max-workers`. You can also try a higher number if you have lots of CPU & memory.

### Mode (b) - creating a cell census from a user-provided list of H5AD files:

- Create a manifest file, in CSV format, containing two columns: dataset_id, h5ad_uri. Example:
  ```csv
  53d208b0-2cfd-4366-9866-c3c6114081bc, /files/53d208b0-2cfd-4366-9866-c3c6114081bc.h5ad
  559ed814-a9c9-4b77-a0e6-7da7b907fe3a, /files/559ed814-a9c9-4b77-a0e6-7da7b907fe3a.h5ad
  5b93b8fc-7c9a-45bd-ad3f-dc883137de30, /files/5b93b8fc-7c9a-45bd-ad3f-dc883137de30.h5ad
  ```
  You can specify a file system path or a URI in the second field
- To create a cell census at `<census_path>`, execute:
  > $ python -m cell_census_builder <census_path> build --manifest <the_manifest_file.csv>

### Other info

There are more options discoverable via the `--help` command line option.

Note on required host resources:

- all H5AD files not on the local disk will be downloaded/cached locally. There must be
  sufficient local file system space. Location of cache can be controlled with the
  environment variable `FSSPEC_CACHE_DIR`
- each H5AD will be read into memory, in its entirety. Sufficient RAM must be present to
  allow for this (and to do so for multiple H5ADs concurrently if you use the `--multi-process` option)
