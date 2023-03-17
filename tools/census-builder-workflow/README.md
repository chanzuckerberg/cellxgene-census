# Cell Census Builder Workflow

This subproject can be used to run a cell-census build using a Docker container and a custom workflow file.

## Instructions

### Build

To build the docker container, `cd` into the parent folder (`tools/`) and run:

```docker build --build-arg=COMMIT_SHA=$(git rev-parse --short HEAD) . -t census-builder```

This will build a Docker container named `census-builder`.

### Prepare

Before running the workflow, make sure that a `data` directory exists on the machine. This can contain any inputs for the builder (e.g. a manifest file and local `h5ad`s), and will also be used to output the built cell census. This folder will also need to contain a `build-census.yaml` file as defined in the next step.


### Create workflow file

In the `data` folder, create a `build-census.yaml` file that contain a workflow that will be executed by the builder. This should also contain all the parameters for the workflow.

Here is an example workflow that runs the builder using a manifest file:

```
census-builder:
  uri:
    /data/cell-census-small/
  verbose:
    true
  commands:
    build:
      manifest:
        /data/manifest-small.csv
      test-disable-dirty-git-check:
        true
```


### Run

Run the builder workflow with:

```docker run --mount type=bind,source="path/to/data",target=/data census-builder```