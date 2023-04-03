#' Read the feature dataset presence matrix.
#'
#' @param census The census object, usually returned by `cellxgene.census::open_soma()`.
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

#' Convenience wrapper around `SOMAExperimentAxisQuery`, to build and execute a
#' query, and return it as a `Seurat` object.
#'
#' @param census The census object, usually returned by `cellxgene.census::open_soma()`.
#' @param organism The organism to query, usually one of `Homo sapiens` or `Mus musculus`
#' @param measurement_name The measurement object to query. Defaults to `RNA`.
#' @param X_name The `X` layer to query. Defaults to `raw`.
#' @param obs_query A `SOMAAxisQuery` for the `obs` axis.
#' @param obs_column_names Columns to fetch for the `obs` data frame.
#' @param var_query A `SOMAAxisQuery` for the `var` axis.
#' @param var_column_names Columns to fetch for the `var` data frame.
#'
#' @return A `Seurat` object containing the sensus slice.
#' @importFrom tiledbsoma SOMAExperimentAxisQuery
#' @export
#'
#' @examples
get_seurat <- function(
    census,
    organism,
    measurement_name = "RNA",
    X_name = "raw",
    obs_query = NULL,
    obs_column_names = NULL,
    var_query = NULL,
    var_column_names = NULL) {
  expt_query <- tiledbsoma::SOMAExperimentAxisQuery$new(
    get_experiment(census, organism),
    measurement_name,
    obs_query = obs_query,
    var_query = var_query
  )
  return(expt_query$to_seurat(
    # TODO: should we allow selection of the seurat 'counts' or 'data' slot?
    c(counts = X_name),
    obs_column_names = obs_column_names,
    var_column_names = var_column_names
  ))
}

#' Get the SOMAExperiment for a named organism
#'
#' @param census The census SOMACollection.
#' @param organism The organism name, e.g. `Homo sapiens`
#'
#' @return a [tiledbsoma::SOMAExperiment] with the requested experiment.
#'
#' @importFrom methods is
#' @importFrom stats setNames
#'
#' @noRd
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
