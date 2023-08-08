#' Open the Census
#'
#' @param census_version The version of the Census, e.g., "stable".
#' @param uri A URI containing the Census SOMA objects to open instead of a
#'            released version. (If supplied, takes precedence over
#'            `census_version`.)
#' @param tiledbsoma_ctx A `tiledbsoma::SOMATileDBContext` built using
#'        `new_SOMATileDBContext_for_census()`. Optional (created automatically)
#'        if using `census_version` and the context does not need to be reused.
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
#' census <- open_soma()
#' as.data.frame(census$get("census_info")$get("summary")$read()$concat())
#' census$close()
open_soma <- function(census_version = "stable", uri = NULL, tiledbsoma_ctx = NULL) {
  if (is.null(uri) || is.null(tiledbsoma_ctx)) {
    description <- get_census_version_description(census_version)
    if (nchar(description$alias) > 0) {
      message(paste("The ", description$alias, " Census release is currently ",
        description$release_build, ". Specify census_version = \"",
        description$release_build,
        "\" in future calls to open_soma() to ensure data consistency.",
        sep = ""
      ))
    }
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

#' Create `SOMATileDBContext` for Census
#' @description Create a SOMATileDBContext suitable for using with `open_soma()`.
#' Typically `open_soma()` creates a context automatically, but one can be created
#' separately in order to set custom configuration options, or to share it between
#' multiple open Census handles.
#'
#' @param census_version_description The result of `get_census_version_description()`
#'        for the desired Census version.
#' @param ... Custom configuration options.
#'
#' @return SOMATileDBContext object for `open_soma()`.
#' @export
#'
#' @examples
#' census_desc <- get_census_version_description("stable")
#' ctx <- new_SOMATileDBContext_for_census(census_desc, "soma.init_buffer_bytes" = paste(4 * 1024**3))
#' census <- open_soma("stable", tiledbsoma_ctx = ctx)
#' census$close()
new_SOMATileDBContext_for_census <- function(census_version_description, ...) {
  # start with default configuration
  cfg <- DEFAULT_TILEDB_CONFIGURATION

  # add vfs.s3.region if specified in census_version_description
  s3_region <- NULL
  if ("soma.s3_region" %in% names(census_version_description) &&
    census_version_description$soma.s3_region != "") {
    cfg <- c(cfg, c("vfs.s3.region" = census_version_description$soma.s3_region))
  }

  # Add unsigned requests
  cfg <- c(cfg, c("vfs.s3.no_sign_request" = "true"))

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
