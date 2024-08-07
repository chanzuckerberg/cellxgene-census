% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/get_helpers.R
\name{get_seurat}
\alias{get_seurat}
\title{Export Census slices to \code{Seurat}}
\usage{
get_seurat(
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
  var_index = "feature_id"
)
}
\arguments{
\item{census}{The census object, usually returned by \code{cellxgene.census::open_soma()}.}

\item{organism}{The organism to query, usually one of \verb{Homo sapiens} or \verb{Mus musculus}}

\item{measurement_name}{The measurement object to query. Defaults to \code{RNA}.}

\item{X_layers}{A named character of \code{X} layers to add to the Seurat assay, where the names are
the names of Seurat slots (\code{counts} or \code{data}) and the values are the names of layers
within \code{X}.}

\item{obs_value_filter}{A SOMA \code{value_filter} across columns in the \code{obs} dataframe, expressed as string.}

\item{obs_coords}{A set of coordinates on the obs dataframe index, expressed in any type or format supported by SOMADataFrame's read() method.}

\item{obs_column_names}{Columns to fetch for the \code{obs} data frame.}

\item{obsm_layers}{Names of arrays in obsm to add as the cell embeddings; pass FALSE to suppress loading in any dimensional reductions.}

\item{var_value_filter}{Same as \code{obs_value_filter} but for \code{var}.}

\item{var_coords}{Same as \code{obs_coords} but for \code{var}.}

\item{var_column_names}{Columns to fetch for the \code{var} data frame.}

\item{var_index}{Name of column in ‘var’ to add as feature names.}
}
\value{
A \code{Seurat} object containing the sensus slice.
}
\description{
Convenience wrapper around \code{SOMAExperimentAxisQuery}, to build and execute a
query, and return it as a \code{Seurat} object.
}
\examples{
\dontrun{
census <- open_soma()
seurat_obj <- get_seurat(
  census,
  organism = "Homo sapiens",
  obs_value_filter = "cell_type == 'leptomeningeal cell'",
  var_value_filter = "feature_id \%in\% c('ENSG00000107317', 'ENSG00000106034')"
)

seurat_obj

census$close()
}
}
