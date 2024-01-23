# README

This package contains code to build and release the Census in the SOMA format, as specified in the
[data schema](https://github.com/chanzuckerberg/cellxgene-census/blob/main/docs/cell_census_schema.md).

This tool is not intended for end-users - it is used by the CELLxGENE team to periodically create and release all
CELLxGENE data in the above format. The remainder of this document is intended for users of the
build package.

Please see the top-level [README](../../README.md) for more information on the Census and
using the Census data.

*NOTE:* this package currently requires Python 3.11, and is only tested on Linux.

## Overview

This package contains sub-modules, each of which automate elements of the Census build and release process.
They are wrapped at the package top-level by by a `__main__` which implements the Census build process,
with standard defaults.

The top-level build can be invoked as follows:

- Create a working directory, e.g., `census-build` or equivalent.
- If any configuration defaults need to be overridden, create a `config.yaml` in the working directory containing the default overrides. *NOTE:* by default you do not need to create a `config.yaml` file -- the defaults are appropriate to build the full Census.
- Run the build as `python -m cellxgene_census_builder your-working_dir`

This will perform four steps (more will be added the future):

- host validation
- build soma
- validate soma
- build reports (eg., summary)

This will result in the following file tree:

```text
working_dir:
    |
    +-- config.yaml        # build config (user provided, read-only)
    +-- state.yaml         # build runtime state (eg., census version tag, etc)
    +-- build-version      # defaults to current date, e.g., 2023-01-20
    |   +-- soma
    |   +-- h5ads
    +-- logs               # log files from various stages
    |   +-- build.log
    |   +-- ...
    +-- reports
        +-- census-summary-VERSION.txt
        +-- census-diff-VERSION.txt
```

## Building and using the Docker container

### Prerequisites

You will need:

- Linux - known to work on Ubuntu 20 and 22, and should work fine on most other (modern) Linux distros
- Docker - [primary installation instructions](https://docs.docker.com/engine/install/ubuntu/#installation-methods) and [important post-install configuration](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)
- Python 3.11

### Build & run

The standard Census build is expected to be done via a Docker container. To build the required image, do a `git pull` to the version you want to use, and do the following to create a docker image called `cellxgene-census-builder`:

```shell
cd tools/cellxgene_census_builder
make image
```

To use the container to build the *full* census, with default options, pick a working directory (e.g., /tmp/census-build), and:

```shell
mkdir /tmp/census-build
docker run -u `id -u`:`id -g` --mount type=bind,source="/tmp/census-build",target='/census-build' cellxgene-census-builder
```

### Pull the Docker image from ECR

Note that a Docker image is pushed to ECR each time `main` gets updated. It is possible to pull an image from the remote repo by running:

```shell
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_REGISTRY.dkr.ecr.us-west-2.amazonaws.com
docker pull $ECR_REGISTRY.dkr.ecr.us-west-2.amazonaws.com/cellxgene-census-builder:latest
```

Alternatively, a tag corresponding to a git tag can be used to pull a specific version.

### Build configuration options

This is primarily for the use of package developers. The defaults are suitable for the standad Census build, and are defined in the `build_state.py` file.

If you need to override a default, create `config.yaml` in the build working directory and specify the overrides. An example `config.yaml` might look like:

```yaml
verbose: 2  # debug level logging
consolidate: false  # disable TileDB consolidation
```

### Commands to cleanup local Docker state on your ec2 instance (while building an image)

Docker keeps around intermediate layers/images and if your machine doesn't have enough memory, you might run into issues. You can blow away these cached layers/images by running the following commands.

```shell
docker system prune
docker rm -f $(docker ps -aq)
docker rmi -f $(docker images -q)
```

## Module-specific notes

### `host_validation` module

Module which provides a set of checks that the current host machine has the requisite capabilities
to build the census (e.g., free disk space). Raises exception (non-zero process exit) if host is
unable to meet base requirements.

Stand-alone usage: `python -m cellxgene_census_builder.host_validation`

### `build_soma` module

Stand-alone use: `python -m cellxgene_census_builder.build_soma ...`

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
- (Optional) Validate the entire Census, re-reading from storage.

Modes of operation:
a) (default) creating the entire "Census" using all files currently in the CELLxGENE repository.
b) creating a smaller "Census" from a user-provided list of files (a "manifest")

#### Mode (a) - creating the full Census from the entire CELLxGENE (public) corpus:

- On a large-memory machine with *ample* free (local) disk (eg, 3/4 TB or more) and swap (1 TB or more)
- To create a Census at `<census_path>`, execute:
  > $ python -m cellxgene_census_builder -mp --max-workers 12 <census_path> build
- Tips:
  - `-v` to view info-level logging during run, or `-v -v` for debug-level logging
  - `--test-first-n <#>` to test build on a subset of datasets
  - `--build-tag $(date +'%Y%m%d_%H%M%S')` to produce non-conflicting census build directories during testing

If you run out of memory, reduce `--max-workers`. You can also try a higher number if you have lots of CPU & memory.

#### Mode (b) - creating a Census from a user-provided list of H5AD files:

- Create a manifest file, in CSV format, containing two columns: dataset_id, h5ad_uri. Example:

  ```csv
  53d208b0-2cfd-4366-9866-c3c6114081bc, /files/53d208b0-2cfd-4366-9866-c3c6114081bc.h5ad
  559ed814-a9c9-4b77-a0e6-7da7b907fe3a, /files/559ed814-a9c9-4b77-a0e6-7da7b907fe3a.h5ad
  5b93b8fc-7c9a-45bd-ad3f-dc883137de30, /files/5b93b8fc-7c9a-45bd-ad3f-dc883137de30.h5ad
  ```

  You can specify a file system path or a URI in the second field
- To create a Census at `<census_path>`, execute:
  > $ python -m cellxgene_census_builder <census_path> build --manifest <the_manifest_file.csv>
