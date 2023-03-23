
# Cell Census of CZ CELLxGENE Discover

<!-- badges: start -->
<!-- badges: end -->


The `CellCensus` package provides an API to facilitate the use of the CZ CELLxGENE Discover Cell Census. For more information about the API and the project visit the [chanzuckerberg/cell-census GitHub repo](https://github.com/chanzuckerberg/cell-census/).

**Status**: Pre-release, under rapid development. Expect API changes.

Also see the [Python API](https://chanzuckerberg.github.io/cell-census/).

## Installation

You can install the development version of CellCensus from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("chanzuckerberg/cell-census/api/r/CellCensus")
print(CellCensus::open_soma())
```

(minimal apt dependencies: r-base cmake git)

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(CellCensus)
## basic example code
```

## For More Help

For more help, please file a issue on the repo, or contact us at <soma@chanzuckerberg.com>

If you believe you have found a security issue, we would appreciate notification. Please send email to <security@chanzuckerberg.com>.
