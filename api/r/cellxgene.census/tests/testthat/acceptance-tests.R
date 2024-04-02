test_that("test_incremental_read_X_mouse", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  organism <- "mus_musculus"

  # Warning that results cannot be concat because it
  # exceeds R's capability to allocate vectors beyond 32bit
  X_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$X$get("raw")$read()$tables()
  expect_true(table_iter_is_ok(X_iter))
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_incremental_read_X_mouse-large-buffer-size", {
  census <- open_soma_latest_for_test(soma.init_buffer_bytes = paste(1 * 1024**3))
  on.exit(census$close(), add = TRUE)

  organism <- "mus_musculus"

  # Warning that results cannot be concat because it
  # exceeds R's capability to allocate vectors beyond 32bit
  X_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$X$get("raw")$read()$tables()
  expect_true(table_iter_is_ok(X_iter))
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_incremental_query_human_brain", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  organism <- "homo_sapiens"
  obs_value_filter <- "tissue == 'brain'"

  query <- tiledbsoma::SOMAExperimentAxisQuery$new(
    experiment = census$get("census_data")$get(organism),
    measurement_name = "RNA",
    obs_query = tiledbsoma::SOMAAxisQuery$new(value_filter = obs_value_filter)
  )

  expect_true(table_iter_is_ok(query$obs()))
  expect_true(table_iter_is_ok(query$var()))
  expect_true(table_iter_is_ok(query$X("raw")$tables()))
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_incremental_query_human_aorta", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  organism <- "homo_sapiens"
  obs_value_filter <- "tissue == 'aorta'"

  query <- tiledbsoma::SOMAExperimentAxisQuery$new(
    experiment = census$get("census_data")$get(organism),
    measurement_name = "RNA",
    obs_query = tiledbsoma::SOMAAxisQuery$new(value_filter = obs_value_filter)
  )

  expect_true(table_iter_is_ok(query$obs()))
  expect_true(table_iter_is_ok(query$var()))
  expect_true(table_iter_is_ok(query$X("raw")$tables()))
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_incremental_query_mouse_brain", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  organism <- "mus_musculus"
  obs_value_filter <- "tissue == 'brain'"

  query <- tiledbsoma::SOMAExperimentAxisQuery$new(
    experiment = census$get("census_data")$get(organism),
    measurement_name = "RNA",
    obs_query = tiledbsoma::SOMAAxisQuery$new(value_filter = obs_value_filter)
  )

  expect_true(table_iter_is_ok(query$obs()))
  expect_true(table_iter_is_ok(query$var()))
  expect_true(table_iter_is_ok(query$X("raw")$tables()))
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_incremental_query_mouse_aorta", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  organism <- "mus_musculus"
  obs_value_filter <- "tissue == 'aorta'"

  query <- tiledbsoma::SOMAExperimentAxisQuery$new(
    experiment = census$get("census_data")$get(organism),
    measurement_name = "RNA",
    obs_query = tiledbsoma::SOMAAxisQuery$new(value_filter = obs_value_filter)
  )

  expect_true(table_iter_is_ok(query$obs()))
  expect_true(table_iter_is_ok(query$var()))
  expect_true(table_iter_is_ok(query$X("raw")$tables()))
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_small-query", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_value_filter = "tissue == 'aorta'"
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_10K-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:10000
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_100K-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:100000
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_250K-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:250000
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_500K-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:500000
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_750K-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:750000
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_1M-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:1e6
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_common-tissue", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_value_filter = "tissue == 'brain'"
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_common-tissue-large-buffer-size", {
  census <- open_soma_latest_for_test(soma.init_buffer_bytes = paste(1 * 1024**3))
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_value_filter = "tissue == 'brain'"
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_common-cell-type", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_value_filter = "cell_type == 'neuron'",
    obs_coords = 1:15000000
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_common-cell-type-large-buffer-size", {
  census <- open_soma_latest_for_test(soma.init_buffer_bytes = paste(1 * 1024**3))
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_value_filter = "cell_type == 'neuron'",
    obs_coords = 1:15000000
  )

  test_seurat(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_seurat_whole-enchilada-large-buffer-size", {
  # SKIP: R is not capable to load into memory
  if (FALSE) {
    census <- open_soma_latest_for_test(soma.init_buffer_bytes = paste(1 * 1024**3))
    on.exit(census$close(), add = TRUE)

    test_args <- list(
      census = census,
      organism = "Homo sapiens",
      measurement_name = "RNA"
    )

    test_seurat(test_args)
  }

  expect_true(TRUE)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_small-query", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_value_filter = "tissue == 'aorta'"
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_10K-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:10000
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_100K-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:100000
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_250K-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:250000
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_500K-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:500000
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_750K-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:750000
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_1M-cells-human", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_coords = 1:1e6
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_common-tissue", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_value_filter = "tissue == 'brain'"
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_common-tissue-large-buffer-size", {
  census <- open_soma_latest_for_test(soma.init_buffer_bytes = paste(1 * 1024**3))
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_value_filter = "tissue == 'brain'"
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_common-cell-type", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_value_filter = "cell_type == 'neuron'",
    obs_coords = 1:15000000
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_common-cell-type-large-buffer-size", {
  census <- open_soma_latest_for_test(soma.init_buffer_bytes = paste(1 * 1024**3))
  on.exit(census$close(), add = TRUE)

  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    obs_value_filter = "cell_type == 'neuron'",
    obs_coords = 1:15000000
  )

  test_sce(test_args)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})

test_that("test_sce_whole-enchilada-large-buffer-size", {
  # SKIP: R is not capable to load into memory
  if (FALSE) {
    census <- open_soma_latest_for_test(soma.init_buffer_bytes = paste(1 * 1024**3))
    on.exit(census$close(), add = TRUE)

    test_args <- list(
      census = census,
      organism = "Homo sapiens",
      measurement_name = "RNA",
    )

    test_sce(test_args)
  }

  expect_true(TRUE)
  gc(verbose = getOption("verbose"), reset = FALSE, full = TRUE)

})
