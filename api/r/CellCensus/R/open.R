DEFAULT_TILEDB_CONFIGURATION <- c(
  "py.init_buffer_bytes" = paste(1 * 1024**3),
  "soma.init_buffer_bytes" = paste(1 * 1024**3)
)

#' Open the Cell Census
#'
#' @param census_version The version of the Census, e.g., "latest"
#' @param uri The URI containing the Census SOMA objects. If specified, takes
#'            precedence over `census_version`.
#'
#' @return Top-level `tiledbsoma::SOMACollection` object
#' @importFrom tiledbsoma SOMACollection
#' @importFrom tiledb tiledb_ctx
#' @export
#'
#' @examples
open_soma <- function(census_version = "latest", uri = "") {
  cfg <- DEFAULT_TILEDB_CONFIGURATION

  if (uri == "") {
    description <- get_census_version_description(census_version)
    uri <- description$soma.uri
    if ("soma.s3_region" %in% names(description) &&
      description$soma.s3_region != "") {
      cfg <- c(cfg, c("vfs.s3.region" = description$soma.s3_region))
    }
  }

  return(tiledbsoma::SOMACollection$new(uri, ctx = tiledb::tiledb_ctx(cfg)))
}
