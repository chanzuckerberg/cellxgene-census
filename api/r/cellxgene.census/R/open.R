#' Open the Census
#'
#' @param census_version The version of the Census, e.g., "stable".
#' @param uri A URI containing the Census SOMA objects to open instead of a
#'            released version. (If supplied, takes precedence over
#'            `census_version`.)
#' @param tiledbsoma_ctx A `tiledbsoma::SOMATileDBContext` built using
#'        `new_SOMATileDBContext_for_census()`. Optional (created automatically)
#'        if using `census_version` and the context does not need to be reused.
#' @param mirror The Census mirror to access; one of `names(get_census_mirror_directory())`.
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
open_soma <- function(
    census_version = "stable",
    uri = NULL,
    tiledbsoma_ctx = NULL,
    mirror = NULL) {
  if (is.null(uri) || is.null(tiledbsoma_ctx)) {
    locator <- resolve_census_locator(census_version, uri, mirror)
    uri <- locator$uri
    if (is.null(tiledbsoma_ctx)) {
      tiledbsoma_ctx <- new_SOMATileDBContext_for_census(NULL, mirror = locator$mirror_info)
    }
  }

  return(tiledbsoma::SOMACollectionOpen(uri, tiledbsoma_ctx = tiledbsoma_ctx))
}

SUPPORTED_PROVIDERS <- c("S3", "file", "unknown")

# helps open_soma resolve uri & mirror information
resolve_census_locator <- function(census_version, uri, mirror) {
  mirror_info <- NULL
  if (is.null(uri)) {
    description <- get_census_version_description(census_version)
    if (nchar(description$alias) > 0) {
      message(paste("The ", description$alias, " Census release is currently ",
        description$release_build, ". Specify census_version = \"",
        description$release_build,
        "\" in future calls to open_soma() to ensure data consistency.",
        sep = ""
      ))
    }
    mirror_info <- get_census_mirror(mirror)
    stopifnot(
      "Unsupported provider for this mirror; try upgrading cellxgene.census package." = (mirror_info$provider %in% SUPPORTED_PROVIDERS)
    )
    if ("relative_uri" %in% names(description) && length(description$relative_uri) > 0) {
      uri <- file.path(mirror_info$base_uri, description$relative_uri, fsep = "/")
    } else {
      # release.json not yet updated for mirrors
      uri <- description$soma.uri
    }
  }
  return(list(uri = uri, mirror_info = mirror_info))
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
#' @param mirror The name of the intended census mirror (or `get_census_mirror_directory()[[name]]`
#'        to save the lookup), or NULL to configure for local file access.
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
new_SOMATileDBContext_for_census <- function(census_version_description, mirror = "default", ...) {
  # NOTE: census_version_description is currently unused, vestigial from before
  # mirror support. But, it might become useful again if provider-specific config
  # needs info from it.

  # start with default configuration
  cfg <- DEFAULT_TILEDB_CONFIGURATION

  # set provider-specific config based on the mirror info
  if (!is.null(mirror)) {
    if (is.character(mirror)) {
      mirror <- get_census_mirror_directory()[[mirror]]
    }
    stopifnot("mirror argument to new_SOMATileDBContext_for_census should be a mirror name or get_census_mirror_directory()[[name]]" = is.list(mirror))
    if (mirror$provider == "S3") {
      cfg <- c(cfg, c("vfs.s3.region" = mirror$region))
      cfg <- c(cfg, c("vfs.s3.no_sign_request" = "true"))
    }
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
