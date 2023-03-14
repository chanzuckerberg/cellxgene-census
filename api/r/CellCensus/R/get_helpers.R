#' Read the feature dataset presence matrix.
#'
#' @param census The census SOMACollection from which to read the presence matrix.
#' @param organism The organism to query, usually one of `Homo sapiens` or `Mus musculus`
#' @param measurement_name The measurement object to query. Defaults to `RNA`.
#'
#' @return a [Matrix::sparseMatrix] with dataset join id & feature join id dimensions,
#'         filled with 1s indicating presence
#' @export
#'
#' @examples
get_presence_matrix <- function(census, organism, measurement_name = "RNA") {
  exp <- get_experiment(census, organism)
  presence <- exp$ms$get(measurement_name)$get("feature_dataset_presence_matrix")
  return(presence$read_sparse_matrix())
}

#' Get the SOMAExperiment for a named organism
#'
#' @param census The census SOMACollection.
#' @param organism The organism name, e.g. `Homo sapiens`
#'
#' @return a [tiledbsoma::SOMAExperiment] with the requested experiment.
#'
#' @NoRd
get_experiment <- function(census, organism) {
  # lower/snake case the organism name to find the experiment name
  exp_name <- tolower(gsub("\\s+", "_", organism))
  census_data <- census$get("census_data")

  stopifnot(setNames(
    exp_name %in% census_data$names(),
    paste("Unknown organism", organism, "- does not exist")
  ))

  exp <- census_data$get(exp_name)

  stopifnot(setNames(
    is(exp, "SOMAExperiment"),
    paste("Unknown organism", organism, "- not a SOMA Experiment")
  ))

  return(exp)
}
