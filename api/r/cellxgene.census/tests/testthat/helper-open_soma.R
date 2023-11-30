# Most test cases use the "latest" version instead of the default "stable" to help
# catch any regressions in the builder/client system. This fixture wraps open_soma()
# to open the "latest" version and reuse a SOMATileDBContext.

tiledbsoma_ctx_latest_for_test <- NULL

open_soma_latest_for_test <- function(...) {
  if (is.null(tiledbsoma_ctx_latest_for_test)) {
    # it's important to initialize tiledbsoma_ctx_latest_for_test "lazily" here
    # rather than at top level since top-level code is evaluated at install time
    # and saved, which leads to confusion with TileDB-R's internal caching of
    # the last-used context.
    tiledbsoma_ctx_latest_for_test <- new_SOMATileDBContext_for_census(
      get_census_version_description("latest"),
      ...
    )
  }
  open_soma("latest", tiledbsoma_ctx = tiledbsoma_ctx_latest_for_test)
}

# A known-good Cell Census version. This may need updating if the version used
# is withdrawn for any reason.
KNOWN_CENSUS_VERSION <- "2023-05-15" # an LTS version
KNOWN_CENSUS_URI <- paste0(
  "s3://cellxgene-census-public-us-west-2/cell-census/",
  KNOWN_CENSUS_VERSION,
  "/soma/"
)
