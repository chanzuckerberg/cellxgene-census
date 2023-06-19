# These are expensive tests and should be run as part of the automated 
# testing framework. They are meant to be run manually via testthat::test_file()

test_that("test_load_axes", {
              
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  
  for (organism in c("homo_sapiens", "mus_musculus")) {
      
      # use subset of columns for speed
      obs_df = census$get("census_data")$get(organism)$obs$read(coords = 1:4, column_names = c("soma_joinid", "cell_type", "tissue"))
      obs_df = as.data.frame(obs_df$concat())
      
      expect_true(nrow(obs_df) > 0)
      
      var_df = census$get("census_data")$get(organism)$ms$get("RNA")$var$read(coords= 1:4)
      var_df = as.data.frame(var_df$concat())
      
      expect_true(nrow(var_df) > 0)
      
      rm(obs_df)
      rm(var_df)
      gc()
      break
  }
  
})

test_that("test_incremental_read", {
              
  # Verify that obs, var and X[raw] can be read incrementally, i.e., in chunks
              
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  
  for (organism in c("homo_sapiens", "mus_musculus")) {
      
      # use subset of columns for speed
      obs_iter <- census$get("census_data")$get(organism)$obs$read(column_names = c("soma_joinid", "cell_type", "tissue"))
      expect_true(table_iter_is_ok(obs_iter))
      
      var_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$var$read()
      expect_true(table_iter_is_ok(var_iter))
      
      # Warning that results cannot be concat because it 
      # exceeds R's capability to allocate vectors beyond 32bit
      expect_warning(X_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$X$get("raw")$read()$tables())
      
      expect_true(table_iter_is_ok(X_iter))
      gc()
      break
  }
  
})

test_that("test_incremental_query", {
  #TODO implement when query$obs() $var() and $X() return iterators, not yet in tiledbsoma
  expect_true(TRUE)
})

test_that("test_seurat_small-query", {
              
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  
  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    X_name = "raw",
    obs_value_filter = "tissue == 'aorta'"
  )
  
  test_seurat(test_args)
  
})

test_that("test_seurat_10K-cells-human", {
              
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  
  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    X_name = "raw",
    obs_coords = 1:10000,
  )
  
  test_seurat(test_args)
                  
})

test_that("test_seurat_100K-cells-human", {
              
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  
  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    X_name = "raw",
    obs_coords = 1:100000,
  )
  
  test_seurat(test_args)
                  
})

test_that("test_seurat_common-cell-type", {
              
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  
  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    X_name = "raw",
    obs_value_filter = "cell_type == 'neuron'",
  )
  
  test_seurat(test_args)
                  
})

test_that("test_seurat_common-tissue", {
              
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  
  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    X_name = "raw",
    obs_value_filter = "tissue == 'brain'",
    obs_coords = NULL,
    obs_column_names = NULL,
    var_value_filter = NULL,
    var_coords = NULL,
    var_column_names = NULL
  )
  
  test_seurat(test_args)
                  
})

test_that("test_seurat_common-tissue-large-buffer-size", {
              
  census <- open_soma_latest_for_test(soma.init_buffer_bytes = paste(4 * 1024**3))
  on.exit(census$close(), add = TRUE)
  
  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    X_name = "raw",
    obs_value_filter = "tissue == 'brain'",
  )
  
  test_seurat(test_args)
                  
})

test_that("test_seurat_whole-enchilada-large-buffer-size", {
              
  census <- open_soma_latest_for_test(soma.init_buffer_bytes = paste(4 * 1024**3))
  on.exit(census$close(), add = TRUE)
  
  test_args <- list(
    census = census,
    organism = "Homo sapiens",
    measurement_name = "RNA",
    X_name = "raw",
  )
  
  test_seurat(test_args)
                  
})
