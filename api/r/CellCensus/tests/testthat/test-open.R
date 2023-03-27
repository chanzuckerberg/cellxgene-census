test_that("open_soma", {
  coll <- open_soma("latest")
  expect_true(startsWith(coll$uri, "s3://cellxgene-data-public/cell-census/"))
  expect_true(coll$exists())
  expect_true(coll$get("census_data")$get("homo_sapiens")$exists())
})

test_that("open_soma latest/default", {
  coll_default <- open_soma()
  coll_latest <- open_soma("latest")
  expect_equal(coll_default$uri, coll_latest$uri)
})

test_that("open_soma with custom context", {
  ctx <- tiledbsoma::SOMATileDBContext$new(config = c("vfs.s3.region" = "bogus"))
  # open_soma should override our bogus vfs.s3.region setting
  coll <- open_soma(tiledbsoma_ctx = ctx)
  expect_true(coll$exists())
})

test_that("open_soma does not sign AWS S3 requests", {
  Sys.setenv(AWS_ACCESS_KEY_ID="fake_id", AWS_SECRET_ACCESS_KEY="fake_key")
  coll <- open_soma("latest")
  expect_true(coll$exists())
  expect_true(coll$get("census_data")$get("homo_sapiens")$exists())
})