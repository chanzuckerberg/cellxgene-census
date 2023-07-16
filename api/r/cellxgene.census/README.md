
# CZ CELLxGENE Discover Census

<!-- badges: start -->
<!-- badges: end -->


The `cellxgene.census` package provides an API to facilitate the use of the CZ CELLxGENE Discover Census. For more information about the API and the project visit the [chanzuckerberg/cellxgene-census GitHub repo](https://github.com/chanzuckerberg/cellxgene-census/).

**Status**: Pre-release, under rapid development. Expect API changes.

Also see the [Python API](https://cellxgene-census.readthedocs.io/).

## Installation

You can install the development version of `cellxgene.census` from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("chanzuckerberg/cellxgene-census/api/r/cellxgene.census")
print(cellxgene.census::open_soma())
```

(minimal apt dependencies: r-base cmake git)

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(cellxgene.census)
## basic example code
```

## For More Help

For more help, please file a issue on the repo, or contact us at <soma@chanzuckerberg.com>

If you believe you have found a security issue, we would appreciate notification. Please send email to <security@chanzuckerberg.com>.

## (internal) vignettes and pkgdown docs

Our vignettes use too much time and memory to build in the routine `R CMD build` procedure (especially in limited-memory GitHub Actions workers used by r-universe, which also doesn't allow build flags like `--no-build-vignettes`). For this reason, the vignettes are stored under [`vignettes_/`](vignettes_) instead of `vignettes/`, to hide them from `R CMD build`.

The vignettes *are* included in our pkgdown documentation site. To build them, the [`update_docs.sh`](update_docs.sh) script locally+temporarily symlinks `vignettes/` to `vignettes_/` so that pkgdown finds them. We then check in the resulting `docs/` folder to git, which the [docsite-build-deploy workflow](../../../.github/workflows/docsite-build-deploy.yml) later copies into the docsite.

1. Ensure R packages up-to-date: tiledbsoma rmarkdown pkgdown
2. Run `api/r/update_docs.sh`
3. Stage the updated pkgdown site into git: `git rm -r --cached api/r/docs/ && git add api/r/docs/`
4. Review and commit
