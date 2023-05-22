#' Open the Cell Census
#'
#' @param census_version The version of the Census, e.g., "latest".
#' @param uri The URI containing the Census SOMA objects. If specified, takes
#'            precedence over `census_version`.
#' @param tiledbsoma_ctx A custom `tiledbsoma::SOMATileDBContext`
#'
#' @return Top-level `tiledbsoma::SOMACollection` object. After use, the census
#'         should be closed to release memory and other resources, usually with
#'         `on.exit(census$close(), add = TRUE)`. Closing the top-level census
#'         will also close all SOMA objects accessed through it.
#' @importFrom tiledbsoma SOMACollection
#' @importFrom tiledbsoma SOMATileDBContext
#' @export
#'
#' @examples
open_soma <- function(census_version = "latest", uri = NULL, tiledbsoma_ctx = NULL) {
  if (is.null(uri) || is.null(tiledbsoma_ctx)) {
    description <- get_census_version_description(census_version)
    if (is.null(uri)) {
      uri <- description$soma.uri
    }
    if (is.null(tiledbsoma_ctx)) {
      tiledbsoma_ctx <- new_SOMATileDBContext_for_census(description)
    }
  }

  return(tiledbsoma::SOMACollectionOpen(uri, tiledbsoma_ctx = tiledbsoma_ctx))
}

DEFAULT_TILEDB_CONFIGURATION <- c(
  "py.init_buffer_bytes" = paste(1 * 1024**3),
  "soma.init_buffer_bytes" = paste(1 * 1024**3)
)

new_SOMATileDBContext_for_census <- function(census_version_description, ...) {
  # start with default configuration
  cfg <- DEFAULT_TILEDB_CONFIGURATION

  # add vfs.s3.region if specified in census_version_description
  s3_region <- NULL
  if ("soma.s3_region" %in% names(census_version_description) &&
    census_version_description$soma.s3_region != "") {
    cfg <- c(cfg, c("vfs.s3.region" = census_version_description$soma.s3_region))
  }

  # merge any additional config from args
  config_args <- list(...)
  for (key in names(config_args)) {
    existing <- names(cfg) == key
    if (sum(existing) > 0) {
      stopifnot(sum(existing) == 1)
      cfg[existing] <- config_args[[key]]
    } else {
      addition <- as.character(config_args[[key]])[1]
      names(addition) <- key
      cfg <- c(cfg, addition)
    }
  }

  # instantiate SOMATileDBContext
  return(tiledbsoma::SOMATileDBContext$new(config = cfg))
}
