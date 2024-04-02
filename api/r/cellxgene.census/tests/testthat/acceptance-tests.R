test_that("test_incremental_read_X_mouse", {
  census <- open_soma_latest_for_test()
  on.exit(census$close(), add = TRUE)

  organism <- "mus_musculus"

  # Warning that results cannot be concat because it
  # exceeds R's capability to allocate vectors beyond 32bit
  X_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$X$get("raw")$read()$tables()
  expect_true(table_iter_is_ok(X_iter))
})