test_that("get_experiment", {
  census <- open_soma()

  cases <- list(
    "mus_musculus" = c("Mus musculus", "mus_musculus"),
    "homo_sapiens" = c("Homo sapiens", "homo_sapiens")
  )

  for (org in names(cases)) {
    uri <- census$get("census_data")$get(org)$uri
    for (alias in c(org, cases[[org]])) {
      expect_equal(get_experiment(census, alias)$uri, uri)
    }
  }

  expect_error(get_experiment(census, "bogus"), "Unknown organism")
})

test_that("get_presence_matrix", {
  census <- open_soma()
  datasets <- as.data.frame(census$get("census_info")$get("datasets")$read())
  for (org in c("homo_sapiens", "mus_musculus")) {
    pm <- get_presence_matrix(census, org)
    expect_true(inherits(pm, "matrixZeroBasedView"))
    expect_s4_class(as.one.based(pm), "sparseMatrix")
    expect_equal(nrow(pm), nrow(datasets))
    expect_equal(
      ncol(pm),
      nrow(census$get("census_data")$get(org)$ms$get("RNA")$var$read(column_names = "soma_joinid"))
    )
    expect_equal(min(pm), 0)
    expect_equal(max(pm), 1)
  }
})

test_that("get_seurat", {
  seurat <- get_seurat(
    open_soma(),
    "Mus musculus",
    obs_value_filter = "tissue_general == 'vasculature'",
    obs_column_names = c("soma_joinid", "cell_type", "tissue", "tissue_general", "assay"),
    var_value_filter = "feature_name %in% c('Gm53058', '0610010K14Rik')",
    var_column_names = c("soma_joinid", "feature_id", "feature_name", "feature_length")
  )

  expect_s4_class(seurat, "Seurat")
  seurat_assay <- seurat[["RNA"]]
  expect_s4_class(seurat_assay, "Assay")
  expect_equal(nrow(seurat_assay), 2)
  expect_gt(ncol(seurat_assay), 0)
  expect_setequal(seurat_assay[[]][, "feature_name"], c("0610010K14Rik", "Gm53058"))
  expect_equal(sum(seurat[[]][, "tissue_general"] == "vasculature"), ncol(seurat_assay))
})

test_that("get_seurat coords", {
  seurat <- get_seurat(
    open_soma(),
    "Mus musculus",
    obs_coords = list(soma_joinid = bit64::as.integer64(0:1000)),
    var_coords = list(soma_joinid = bit64::as.integer64(0:2000))
  )
  expect_equal(nrow(seurat[[]]), 1001) # obs dataframe
  seurat_assay <- seurat[["RNA"]]
  expect_equal(nrow(seurat_assay[[]]), 2001) # var dataframe
  # NOTE: seurat assay matrix is var x obs, not obs x var
  expect_equal(nrow(seurat_assay), 2001)
  expect_equal(ncol(seurat_assay), 1001)
})

test_that("get_seurat allows missing obs or var filter", {
  census <- open_soma()

  obs_value_filter <- "tissue == 'aorta'"

  obs_query <- tiledbsoma::SOMAAxisQuery$new(
    value_filter = obs_value_filter
  )
  seurat <- get_seurat(census, "Mus musculus",
    obs_value_filter = obs_value_filter,
    obs_column_names = c("soma_joinid"),
    var_column_names = c("soma_joinid")
  )
  control_query <- tiledbsoma::SOMAExperimentAxisQuery$new(
    get_experiment(census, "Mus musculus"),
    "RNA",
    obs_query = obs_query
  )
  expect_equal(ncol(seurat[["RNA"]]), control_query$n_obs)
  expect_equal(nrow(seurat[["RNA"]]), control_query$n_vars)

  seurat <- get_seurat(census, "Mus musculus",
    obs_coords = list(soma_joinid = bit64::as.integer64(0:10000)),
    var_value_filter = "feature_id == 'ENSMUSG00000069581'"
  )
  expect_equal(ncol(seurat[["RNA"]]), 10001)
  expect_equal(nrow(seurat[["RNA"]]), 1)
})
