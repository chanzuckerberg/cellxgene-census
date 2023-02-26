test_that("open_soma", {
  coll <- open_soma("2023-02-13")
  expect_equal(coll$uri, "s3://cellxgene-data-public/cell-census/2023-02-13/soma/")
  expect_true(coll$exists())
  expect_true(coll$get("census_data")$get("homo_sapiens")$exists())
})

test_that("open_soma latest/default", {
  coll_default <- open_soma()
  coll_latest <- open_soma("latest")
  expect_equal(coll_default$uri, coll_latest$uri)
})
