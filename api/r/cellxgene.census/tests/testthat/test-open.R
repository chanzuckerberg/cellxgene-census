test_that("open_soma", {
  coll <- open_soma("latest")
  on.exit(coll$close(), add = TRUE)
  expect_true(coll$is_open())
  expect_equal(coll$mode(), "READ")
  expect_true(startsWith(coll$uri, "s3://cellxgene-data-public/cell-census/"))
  expect_true(coll$exists())
  expect_true(coll$get("census_data")$get("homo_sapiens")$exists())
})

test_that("open_soma latest/default", {
  coll_default <- open_soma()
  on.exit(coll_default$close(), add = TRUE)
  coll_latest <- open_soma("latest")
  on.exit(coll_latest$close(), add = TRUE)
  expect_equal(coll_default$uri, coll_latest$uri)
})

test_that("open_soma with custom context", {
  ctx <- tiledbsoma::SOMATileDBContext$new(config = c("vfs.s3.region" = "bogus"))
  # open_soma should override our bogus vfs.s3.region setting
  coll <- open_soma(tiledbsoma_ctx = ctx)
  on.exit(coll$close(), add = TRUE)
  expect_true(coll$exists())
})
