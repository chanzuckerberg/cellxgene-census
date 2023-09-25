test_that("get_census_version_description", {
  desc <- get_census_version_description(KNOWN_CENSUS_VERSION)
  expect_equal(desc$release_build, KNOWN_CENSUS_VERSION)
  expect_equal(desc$soma.uri, KNOWN_CENSUS_URI)

  # alias resolution
  desc <- get_census_version_description("stable")
  expect_true(is.list(desc))
  expect_true(is.character(desc$release_build))
  expect_true(is.character(desc$soma.uri))
})

test_that("get_census_version_directory", {
  df <- get_census_version_directory()
  expect_true(is.data.frame(df))
  desc <- as.list(df[KNOWN_CENSUS_VERSION, ])
  expect_equal(desc$release_build, KNOWN_CENSUS_VERSION)
  expect_equal(desc$soma.uri, KNOWN_CENSUS_URI)
})
