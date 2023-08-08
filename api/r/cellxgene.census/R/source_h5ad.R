#' Locate source h5ad file for a dataset.
#'
#' @param dataset_id The dataset_id of interest.
#' @param census_version The desired Census version.
#' @param census An open Census handle for `census_version`. If not provided, then
#'               it will be opened and closed automatically; but it's more efficient
#'               to reuse a handle if calling `get_source_h5ad_uri()` multiple times.
#'
#' @return A list with `uri` and optional `s3_region`.
#' @importFrom httr parse_url
#' @importFrom httr build_url
#' @export
#'
#' @examples
#' get_source_h5ad_uri("0895c838-e550-48a3-a777-dbcd35d30272")
get_source_h5ad_uri <- function(dataset_id, census_version = "stable", census = NULL) {
  description <- get_census_version_description(census_version)
  if (is.null(census)) {
    census <- open_soma(
      census_version,
      uri = description$soma.uri,
      tiledbsoma_ctx = new_SOMATileDBContext_for_census(description)
    )
    on.exit(census$close(), add = TRUE)
  }

  dataset <- as.data.frame(
    census$get("census_info")$get("datasets")$read(
      value_filter = paste("dataset_id == '", dataset_id, "'", sep = "")
    )$concat()
  )
  stopifnot("Unknown dataset_id" = nrow(dataset) == 1)
  dataset <- as.list(dataset[1, ])

  # append dataset filename to census base URI for h5ad's
  url <- httr::parse_url(description$h5ads.uri)
  if (!endsWith(paste(url$path, collapse = ""), "/")) {
    # surprised httr::build_url doesn't automatically deal with trailing slash
    url$path <- paste(url$path, "/", sep = "")
  }
  url$path <- paste(url$path, dataset$dataset_h5ad_path, sep = "")

  return(list(
    uri = httr::build_url(url),
    s3_region = description$soma.s3_region
  ))
}

#' Download source H5AD to local file name.
#'
#' @param dataset_id The dataset_id of interest.
#' @param file Local file name to store H5AD file.
#' @param overwrite TRUE to allow overwriting an existing file.
#' @param census_version The desired Census version.
#' @param census An open Census handle for `census_version`. If not provided, then
#'               it will be opened and closed automatically; but it's more efficient
#'               to reuse a handle if calling `download_source_h5ad()` multiple times.
#'
#' @importFrom httr parse_url
#' @importFrom aws.s3 save_object
#' @export
#'
#' @examples
#' download_source_h5ad("0895c838-e550-48a3-a777-dbcd35d30272", "/tmp/data.h5ad", overwrite = TRUE)
download_source_h5ad <- function(dataset_id, file, overwrite = FALSE,
                                 census_version = "stable", census = NULL) {
  stopifnot("specify local filename, not directory" = !dir.exists(file))
  loc <- get_source_h5ad_uri(dataset_id, census_version = census_version, census = census)
  url <- httr::parse_url(loc$uri)
  stopifnot("only S3 sources supported" = url$scheme == "s3")

  aws.s3::save_object(
    bucket = url$hostname, object = url$path,
    region = loc$s3_region,
    file = file,
    overwrite = overwrite
  )

  invisible()
}
