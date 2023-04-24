DEFAULT_TILEDB_CONFIGURATION <- c(
  "py.init_buffer_bytes" = paste(1 * 1024**3),
  "soma.init_buffer_bytes" = paste(1 * 1024**3)
)

#' Open the Cell Census
#'
#' @param census_version The version of the Census, e.g., "latest".
#' @param uri The URI containing the Census SOMA objects. If specified, takes
#'            precedence over `census_version`.
#' @param tiledbsoma_ctx A custom `tiledbsoma::SOMATileDBContext`
#'
#' @return Top-level `tiledbsoma::SOMACollection` object
#' @importFrom tiledbsoma SOMACollection
#' @importFrom tiledbsoma SOMATileDBContext
#' @export
#'
#' @examples
open_soma <- function(census_version = "latest", uri = NULL, tiledbsoma_ctx = NULL) {
  s3_region <- NULL

  if (is.null(uri)) {
    description <- get_census_version_description(census_version)
    uri <- description$soma.uri
    if ("soma.s3_region" %in% names(description) &&
      description$soma.s3_region != "") {
      s3_region <- description$soma.s3_region
    }
  }

  cfg <- DEFAULT_TILEDB_CONFIGURATION
  cfg <- c(cfg, c("vfs.s3.no_sign_request" = "true"))
  if (is.null(tiledbsoma_ctx)) {
    if (!is.null(s3_region)) {
      cfg <- c(cfg, c("vfs.s3.region" = description$soma.s3_region))
    }
  } else {
    # FIXME: we should use something like SOMATileDBContext$replace() (yet to
    # exist) in case the user context has other important fields besides config
    cfg <- as.vector(tiledb::config(tiledbsoma_ctx$context()))
    if (!is.null(s3_region)) {
      cfg["vfs.s3.region"] <- s3_region
    }
  }
  tiledbsoma_ctx <- tiledbsoma::SOMATileDBContext$new(config = cfg)

  return(tiledbsoma::SOMACollectionOpen(uri, tiledbsoma_ctx = tiledbsoma_ctx))
}
