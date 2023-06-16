# These are expensive tests and should be run as part of the automated 
# testing framework. They are meant to be run manually via testthat::test_file()

test_that("test_load_axes", {
              
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  
  for (oganism in c("homo_sapiens", "mus_musculus")) {
      
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
  }
  
})

test_that("test_incremental_read", {
              
  # Verify that obs, var and X[raw] can be read incrementally, i.e., in chunks
              
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)
  
  for (oganism in c("homo_sapiens", "mus_musculus")) {
      
      # use subset of columns for speed
      obs_iter <- census$get("census_data")$get(organism)$obs$read(column_names = c("soma_joinid", "cell_type", "tissue"))
      expect_true(table_iter_is_ok(obs_iter))
      
      var_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$var$read()
      expect_true(table_iter_is_ok(var_iter))
      
      expect_warning(X_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$X$get("raw")$read()$tables(),
                    "Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'")
      expect_true(table_iter_is_ok(X_iter))
      gc()
  }
  
})

test_that("test_incremental_query", {
  #TODO implement when query$obs() $var() and $X() return iterators, not yet in tiledbsoma
  expect_true(TRUE)
})


