CELL_CENSUS_RELEASE_DIRECTORY_URL <- "https://census.cellxgene.cziscience.com/cellxgene-census/v1/release.json"


#' Get release description for given census version
#'
#' @param census_version The census version name.
#'
#' @return List with the release location and metadata
#' @export
#'
#' @examples
get_census_version_description <- function(census_version) {
  census_directory <- get_census_version_directory()
  description <- census_directory[census_version, ]
  if (nrow(description) == 0) {
    stop(paste(
      "The", census_version, "Census version is not valid.",
      "Use get_census_version_directory() to retrieve available versions."
    ))
  }
  ans <- as.list(description)
  ans$census_version <- census_version
  return(ans)
}

#' Get the directory of cell census releases currently available
#'
#' @return Data frame of available cell census releases, including location and
#'   metadata.
#' @importFrom jsonlite fromJSON
#' @export
#'
#' @examples
get_census_version_directory <- function() {
  raw <- jsonlite::fromJSON(CELL_CENSUS_RELEASE_DIRECTORY_URL)

  # Resolve all aliases for easier use
  for (field in names(raw)) {
    points_at <- raw[[field]]
    while (is.character(points_at)) {
      points_at <- raw[[points_at]]
    }
    points_at[["alias"]] <- if (is.character(raw[[field]])) field else ""
    # ^ that line actually does NOT modify `raw` because points_at is a copy;
    # https://www.oreilly.com/library/view/r-in-a/9781449358204/ch05s05.html
    raw[[field]] <- points_at
  }

  # Replace NULLs with empty string to facilitate data frame conversion
  raw <- simple_rapply(raw, function(x) ifelse(is.null(x), "", x))

  # Convert nested list to data frame
  df <- do.call(rbind, lapply(raw, as.data.frame))
  rownames(df) <- names(raw)
  return(df)
}

# https://stackoverflow.com/a/38950304
simple_rapply <- function(x, fn) {
  if (is.list(x)) {
    lapply(x, simple_rapply, fn)
  } else {
    fn(x)
  }
}
