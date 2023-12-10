CELL_CENSUS_RELEASE_DIRECTORY_URL <- "https://census.cellxgene.cziscience.com/cellxgene-census/v1/release.json"
CELL_CENSUS_MIRRORS_DIRECTORY_URL <- "https://census.cellxgene.cziscience.com/cellxgene-census/v1/mirrors.json"


#' Get release description for a Census version
#'
#' @param census_version The census version name.
#'
#' @return List with the release location and metadata
#' @export
#'
#' @examples
#' as.data.frame(get_census_version_description("stable"))
get_census_version_description <- function(census_version) {
  census_directory <- get_census_version_directory()
  if (!(census_version %in% rownames(census_directory))) {
    stop(paste(
      "The", census_version, "Census version is not valid.",
      "Use get_census_version_directory() to retrieve available versions."
    ))
  }
  description <- census_directory[census_version, ]
  ans <- as.list(description)
  ans$census_version <- census_version
  return(ans)
}

#' Get the directory of Census releases currently available
#'
#' @return Data frame of available cell census releases, including location and
#'   metadata.
#' @importFrom jsonlite fromJSON
#' @importFrom dplyr bind_rows
#' @export
#'
#' @examples
#' get_census_version_directory()
get_census_version_directory <- function() {
  raw <- resolve_aliases(jsonlite::fromJSON(CELL_CENSUS_RELEASE_DIRECTORY_URL))

  # Replace NULLs with empty string to facilitate data frame conversion
  raw <- simple_rapply(raw, function(x) ifelse(is.null(x), "", x))

  # Convert nested list to data frame
  df <- do.call(dplyr::bind_rows, lapply(raw, as.data.frame))
  rownames(df) <- names(raw)
  return(df)
}

#' Get the directory of Census mirrors currently available
#'
#' @return Nested list with information about available mirrors
#' @importFrom jsonlite fromJSON
#' @export
#'
#' @examples
#' get_census_mirror_directory()
get_census_mirror_directory <- function() {
  return(resolve_aliases(jsonlite::fromJSON(CELL_CENSUS_MIRRORS_DIRECTORY_URL)))
}

#' Get locator information about a Census mirror
#'
#' @param mirror Name of the mirror.
#' @return List with mirror information
#' @export
#'
#' @examples
#' get_census_mirror("AWS-S3-us-west-2")
get_census_mirror <- function(mirror) {
  if (is.null(mirror)) {
    mirror <- "default"
  }
  mirrors <- get_census_mirror_directory()
  stopifnot("Unknown Census mirror; use get_census_mirror_directory() to retrieve available mirrors." = (mirror %in% names(mirrors)))
  return(mirrors[[mirror]])
}


# https://stackoverflow.com/a/38950304
simple_rapply <- function(x, fn) {
  if (is.list(x)) {
    lapply(x, simple_rapply, fn)
  } else {
    fn(x)
  }
}

# Given a nested list, top-level character values are assumed to correspond to
# the names of other list items; replace each such character value with the value
# corresponding to the name.
resolve_aliases <- function(obj) {
  # Resolve all aliases for easier use
  for (field in names(obj)) {
    points_at <- obj[[field]]
    while (is.character(points_at)) {
      points_at <- obj[[points_at]]
    }
    points_at[["alias"]] <- if (is.character(obj[[field]])) field else ""
    # ^ that line actually does NOT modify `obj` because points_at is a copy;
    # https://www.oreilly.com/library/view/r-in-a/9781449358204/ch05s05.html
    obj[[field]] <- points_at
  }
  return(obj)
}
