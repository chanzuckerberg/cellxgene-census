test_that("get_experiment", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

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
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  datasets <- as.data.frame(census$get("census_info")$get("datasets")$read()$concat())
  for (org in c("homo_sapiens", "mus_musculus")) {
    pm <- get_presence_matrix(census, org)
    expect_true("matrixZeroBasedView" %in% class(pm))
    pm1 <- pm$get_one_based_matrix()
    expect_s4_class(pm1, "sparseMatrix")
    expect_equal(pm$nrow(), nrow(datasets))
    expect_equal(
      pm$ncol(),
      nrow(census$get("census_data")$get(org)$ms$get("RNA")$var$read(column_names = "soma_joinid")$concat())
    )
    expect_equal(min(pm1), 0)
    expect_equal(max(pm1), 1)
  }
})

test_that("get_seurat", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  seurat <- get_seurat(
    census,
    "Mus musculus",
    obs_value_filter = "tissue_general == 'vasculature'",
    obs_column_names = c("soma_joinid", "cell_type", "tissue", "tissue_general", "assay"),
    # TODO when obsm_layer(s) are available in public census:
    # obsm_layers = c("scvi"),
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

  # reductions should be empty on default setting of obsm_layers:
  expect_equal(length(Reductions(seurat)), 0)
})

test_that("get_seurat coords", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  seurat <- get_seurat(
    census,
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
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

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

test_that("get_seurat X_layers allows slot selection", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  # default: raw X layer gets loaded into counts (which Seurat implicitly copies into data)
  seurat <- get_seurat(
    census,
    "Mus musculus",
    obs_value_filter = "tissue_general == 'vasculature'",
    obs_column_names = c("soma_joinid", "cell_type", "tissue", "tissue_general", "assay"),
    var_value_filter = "feature_name %in% c('Gm53058', '0610010K14Rik')",
    var_column_names = c("soma_joinid", "feature_id", "feature_name", "feature_length")
  )
  expect_equal(nrow(seurat$RNA@counts), 2)
  expect_equal(nrow(seurat$RNA@data), 2)

  # expressly load raw X data into data; then counts is empty
  seurat <- get_seurat(
    census,
    "Mus musculus",
    obs_value_filter = "tissue_general == 'vasculature'",
    obs_column_names = c("soma_joinid", "cell_type", "tissue", "tissue_general", "assay"),
    var_value_filter = "feature_name %in% c('Gm53058', '0610010K14Rik')",
    var_column_names = c("soma_joinid", "feature_id", "feature_name", "feature_length"),
    X_layers = c(counts = NULL, data = "raw")
  )
  expect_equal(nrow(seurat$RNA@counts), 0)
  expect_equal(nrow(seurat$RNA@data), 2)
})

test_that("get_single_cell_experiment", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  sce <- get_single_cell_experiment(
    census,
    "Mus musculus",
    obs_value_filter = "tissue_general == 'vasculature'",
    obs_column_names = c("soma_joinid", "cell_type", "tissue", "tissue_general", "assay"),
    # TODO when obsm_layer(s) are available in public census:
    # obsm_layers = c("scvi"),
    var_value_filter = "feature_name %in% c('Gm53058', '0610010K14Rik')",
    var_column_names = c("soma_joinid", "feature_id", "feature_name", "feature_length")
  )

  expect_s4_class(sce, "SingleCellExperiment")
  expect_equal(nrow(sce), 2)
  expect_gt(ncol(sce), 0)

  expect_s4_class(SingleCellExperiment::colData(sce), "DFrame")
  expect_s4_class(SingleCellExperiment::rowData(sce), "DFrame")

  expect_equal(nrow(SingleCellExperiment::rowData(sce)), 2)
  expect_gt(ncol(SingleCellExperiment::colData(sce)), 0)

  expect_setequal(SingleCellExperiment::rowData(sce)[, "feature_name"], c("0610010K14Rik", "Gm53058"))
  expect_equal(sum(SingleCellExperiment::colData(sce)[, "tissue_general"] == "vasculature"), ncol(sce))

  # reductions should be empty on default setting of obsm_layers:
  expect_equal(length(SingleCellExperiment::reducedDims(sce)), 0)
})

test_that("get_single_cell_experiment coords", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  sce <- get_single_cell_experiment(
    census,
    "Mus musculus",
    obs_coords = list(soma_joinid = bit64::as.integer64(0:1000)),
    var_coords = list(soma_joinid = bit64::as.integer64(0:2000))
  )
  expect_equal(nrow(SingleCellExperiment::colData(sce)), 1001)
  expect_equal(nrow(SingleCellExperiment::rowData(sce)), 2001)
  expect_equal(nrow(SummarizedExperiment::assay(sce)), 2001)
  expect_equal(ncol(SummarizedExperiment::assay(sce)), 1001)
})

test_that("get_single_cell_experiment allows missing obs or var filter", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  obs_value_filter <- "tissue == 'aorta'"

  obs_query <- tiledbsoma::SOMAAxisQuery$new(
    value_filter = obs_value_filter
  )
  sce <- get_single_cell_experiment(
    census,
    "Mus musculus",
    obs_value_filter = obs_value_filter,
    obs_column_names = c("soma_joinid"),
    var_column_names = c("soma_joinid")
  )
  control_query <- tiledbsoma::SOMAExperimentAxisQuery$new(
    get_experiment(census, "Mus musculus"),
    "RNA",
    obs_query = obs_query
  )
  expect_equal(ncol(sce), control_query$n_obs)
  expect_equal(nrow(sce), control_query$n_vars)

  sce <- get_single_cell_experiment(census, "Mus musculus",
    obs_coords = list(soma_joinid = bit64::as.integer64(0:10000)),
    var_value_filter = "feature_id == 'ENSMUSG00000069581'"
  )
  expect_equal(ncol(sce), 10001)
  expect_equal(nrow(sce), 1)
})

test_that("get_single_cell_experiment X_layers allows slot selection", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  # default: raw X layer gets loaded into counts
  sce <- get_single_cell_experiment(
    census,
    "Mus musculus",
    obs_value_filter = "tissue_general == 'vasculature'",
    obs_column_names = c("soma_joinid", "cell_type", "tissue", "tissue_general", "assay"),
    var_value_filter = "feature_name %in% c('Gm53058', '0610010K14Rik')",
    var_column_names = c("soma_joinid", "feature_id", "feature_name", "feature_length")
  )
  expect_equal(nrow(SummarizedExperiment::assay(sce)), 2)

  # load raw X into "counts" and "normcounts"
  sce <- get_single_cell_experiment(
    census,
    "Mus musculus",
    obs_value_filter = "tissue_general == 'vasculature'",
    obs_column_names = c("soma_joinid", "cell_type", "tissue", "tissue_general", "assay"),
    var_value_filter = "feature_name %in% c('Gm53058', '0610010K14Rik')",
    var_column_names = c("soma_joinid", "feature_id", "feature_name", "feature_length"),
    X_layers = c(counts = "raw", normcounts = "raw")
  )

  expect_equal(nrow(SummarizedExperiment::assay(sce, "counts")), 2)
  expect_equal(nrow(SummarizedExperiment::assay(sce, "normcounts")), 2)
  expect_equal(
    sum(SummarizedExperiment::assay(sce, "counts") != SummarizedExperiment::assay(sce, "normcounts")),
    0
  )
})
