#' Read the feature dataset presence matrix.
#'
#' @param census The census object from `cellxgene.census::open_soma()`.
#' @param organism The organism to query, usually one of `Homo sapiens` or `Mus musculus`
#' @param measurement_name The measurement object to query. Defaults to `RNA`.
#'
#' @return A `tiledbsoma::matrixZeroBasedView` object with dataset join id & feature
#'         join id dimensions, filled with 1s indicating presence. The sparse matrix
#'         is accessed with zero-based indexes since the join id's may be zero.
#' @export
#'
#' @examples
#' census <- open_soma()
#' on.exit(census$close(), add = TRUE)
#' print(get_presence_matrix(census, "Homo sapiens")$dim())
get_presence_matrix <- function(census, organism, measurement_name = "RNA") {
  exp <- get_experiment(census, organism)
  presence <- exp$ms$get(measurement_name)$get("feature_dataset_presence_matrix")
  return(presence$read()$sparse_matrix(zero_based = TRUE)$concat())
}


#' Export Census slices to `Seurat`
#'
#' Convenience wrapper around `SOMAExperimentAxisQuery`, to build and execute a
#' query, and return it as a `Seurat` object.
#'
#' @param census The census object, usually returned by `cellxgene.census::open_soma()`.
#' @param organism The organism to query, usually one of `Homo sapiens` or `Mus musculus`
#' @param measurement_name The measurement object to query. Defaults to `RNA`.
#' @param X_layers A named character of `X` layers to add to the Seurat assay, where the names are
#'        the names of Seurat slots (`counts` or `data`) and the values are the names of layers
#'        within `X`.
#' @param obs_value_filter A SOMA `value_filter` across columns in the `obs` dataframe, expressed as string.
#' @param obs_coords A set of coordinates on the obs dataframe index, expressed in any type or format supported by SOMADataFrame's read() method.
#' @param obs_column_names Columns to fetch for the `obs` data frame.
#' @param obsm_layers Names of arrays in obsm to add as the cell embeddings; pass FALSE to suppress loading in any dimensional reductions.
#' @param var_value_filter Same as `obs_value_filter` but for `var`.
#' @param var_coords Same as `obs_coords` but for `var`.
#' @param var_column_names Columns to fetch for the `var` data frame.
#' @param var_index Name of column in ‘var’ to add as feature names.
#'
#' @return A `Seurat` object containing the sensus slice.
#' @importFrom tiledbsoma SOMAExperimentAxisQuery
#' @export
#'
#' @examples
#' \dontrun{
#' census <- open_soma()
#' seurat_obj <- get_seurat(
#'   census,
#'   organism = "Homo sapiens",
#'   obs_value_filter = "cell_type == 'leptomeningeal cell'",
#'   var_value_filter = "feature_id %in% c('ENSG00000107317', 'ENSG00000106034')"
#' )
#'
#' seurat_obj
#'
#' census$close()
#' }
get_seurat <- function(
    census,
    organism,
    measurement_name = "RNA",
    X_layers = c(counts = "raw", data = NULL),
    obs_value_filter = NULL,
    obs_coords = NULL,
    obs_column_names = NULL,
    obsm_layers = FALSE,
    var_value_filter = NULL,
    var_coords = NULL,
    var_column_names = NULL,
    var_index = "feature_id") {
  stopifnot(
    "R package 'Seurat' is not installed." = require("Seurat", quietly = T)
  )

  expt_query <- tiledbsoma::SOMAExperimentAxisQuery$new(
    get_experiment(census, organism),
    measurement_name,
    obs_query = tiledbsoma::SOMAAxisQuery$new(value_filter = obs_value_filter, coords = obs_coords),
    var_query = tiledbsoma::SOMAAxisQuery$new(value_filter = var_value_filter, coords = var_coords)
  )

  return(expt_query$to_seurat(
    X_layers = X_layers,
    obs_column_names = obs_column_names,
    obsm_layers = obsm_layers,
    var_column_names = var_column_names,
    var_index = var_index
  ))
}

#' Export Census slices to `SingleCellExperiment`
#'
#' Convenience wrapper around `SOMAExperimentAxisQuery`, to build and execute a
#' query, and return it as a `SingleCellExperiment` object.
#'
#' @param census The census object, usually returned by `cellxgene.census::open_soma()`.
#' @param organism The organism to query, usually one of `Homo sapiens` or `Mus musculus`
#' @param measurement_name The measurement object to query. Defaults to `RNA`.
#' @param X_layers A character vector of X layers to add as assays in
#'        the main experiment; may optionally be named to set the
#'        name of the resulting assay (eg. ‘X_layers = c(counts =
#'        "raw")’ will load in X layer “‘raw’” as assay “‘counts’”);
#'        by default, loads in all X layers
#' @param obs_value_filter A SOMA `value_filter` across columns in the `obs` dataframe, expressed as string.
#' @param obs_coords A set of coordinates on the obs dataframe index, expressed in any type or format supported by SOMADataFrame's read() method.
#' @param obs_column_names Columns to fetch for the `obs` data frame.
#' @param obsm_layers Names of arrays in obsm to add as the cell embeddings; pass FALSE to suppress loading in any dimensional reductions.
#' @param var_value_filter Same as `obs_value_filter` but for `var`.
#' @param var_coords Same as `obs_coords` but for `var`.
#' @param var_column_names Columns to fetch for the `var` data frame.
#' @param var_index Name of column in ‘var’ to add as feature names.
#'
#' @return A `SingleCellExperiment` object containing the sensus slice.
#' @importFrom tiledbsoma SOMAExperimentAxisQuery
#' @export
#'
#' @examples
#' \dontrun{
#' census <- open_soma()
#' sce_obj <- get_single_cell_experiment(
#'   census,
#'   organism = "Homo sapiens",
#'   obs_value_filter = "cell_type == 'leptomeningeal cell'",
#'   var_value_filter = "feature_id %in% c('ENSG00000107317', 'ENSG00000106034')"
#' )
#'
#' sce_obj
#'
#' census$close()
#' }
get_single_cell_experiment <- function(
    census,
    organism,
    measurement_name = "RNA",
    X_layers = c(counts = "raw"),
    obs_value_filter = NULL,
    obs_coords = NULL,
    obs_column_names = NULL,
    obsm_layers = FALSE,
    var_value_filter = NULL,
    var_coords = NULL,
    var_column_names = NULL,
    var_index = "feature_id") {
  stopifnot(
    "R package 'SingleCellExperiment' is not installed." = require("SingleCellExperiment", quietly = T)
  )

  expt_query <- tiledbsoma::SOMAExperimentAxisQuery$new(
    get_experiment(census, organism),
    measurement_name,
    obs_query = tiledbsoma::SOMAAxisQuery$new(value_filter = obs_value_filter, coords = obs_coords),
    var_query = tiledbsoma::SOMAAxisQuery$new(value_filter = var_value_filter, coords = var_coords)
  )

  return(expt_query$to_single_cell_experiment(
    X_layers = X_layers,
    obs_column_names = obs_column_names,
    obsm_layers = obsm_layers,
    var_column_names = var_column_names,
    var_index = var_index
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
