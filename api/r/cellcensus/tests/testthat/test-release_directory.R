test_that("get_census_version_description", {
  desc <- get_census_version_description("2023-02-13")
  expect_equal(desc$release_build, "2023-02-13")
  expect_equal(desc$soma.uri, "s3://cellxgene-data-public/cell-census/2023-02-13/soma/")

  # alias resolution
  desc <- get_census_version_description("latest")
  expect_true(is.list(desc))
  expect_true(is.character(desc$release_build))
  expect_true(is.character(desc$soma.uri))
})

test_that("get_census_version_directory", {
  df <- get_census_version_directory()
  expect_true(is.data.frame(df))
  desc <- as.list(df["2023-02-13",])
  expect_equal(desc$release_build, "2023-02-13")
  expect_equal(desc$soma.uri, "s3://cellxgene-data-public/cell-census/2023-02-13/soma/")
})
