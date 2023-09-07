# Installation

This Dockerfile & script are used to test the installation instructions for our `cellxgene.census` R package in a clean environment (in contrast to developer machines which tend to have a lot of unrelated packages and other miscellaneous state). Simply:

```shell
docker build -t cellxgene_census_r_install_test api/r/cellxgene.census/tests/installation
```

And verify successful completion, which unfortunately may take 30+ minutes building all the dependencies.

The [install.R](install.R) script should be kept consistent with the user-facing installation instructions in the [package readme](../../README.md).
