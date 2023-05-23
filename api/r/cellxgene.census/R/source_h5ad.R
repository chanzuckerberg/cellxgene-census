#' Locate source h5ad file for a dataset.
#'
#' @param dataset_id The dataset_id of interest.
#' @param census_version The census version.
#'
#' @return A list with `uri` and optional `s3_region`.
#' @importFrom httr parse_url
#' @importFrom httr build_url
#' @export
#'
#' @examples
get_source_h5ad_uri <- function(dataset_id, census_version = "latest") {
  description <- get_census_version_description(census_version)
  census <- open_soma(
    census_version,
    uri = description$soma.uri,
    tiledbsoma_ctx = tiledbsoma::SOMATileDBContext$new(
      config = c("vfs.s3.region" = description$soma.s3_region, "vfs.s3.no_sign_request" = "true")
    )
  )
  on.exit(census$close(), add = TRUE)

  dataset <- as.data.frame(
    census$get("census_info")$get("datasets")$read(
      value_filter = paste("dataset_id == '", dataset_id, "'", sep = "")
    )
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
#' @param census_version The census version.
#'
#' @importFrom httr parse_url
#' @importFrom aws.s3 save_object
#' @export
#'
#' @examples
download_source_h5ad <- function(dataset_id, file, overwrite = FALSE,
                                 census_version = "latest") {
  stopifnot("specify local filename, not directory" = !dir.exists(file))
  loc <- get_source_h5ad_uri(dataset_id, census_version = census_version)
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
