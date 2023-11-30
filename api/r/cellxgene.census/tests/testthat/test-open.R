test_that("open_soma", {
  coll <- open_soma("latest")
  on.exit(coll$close(), add = TRUE)
  expect_true(coll$is_open())
  expect_equal(coll$mode(), "READ")
  expect_true(startsWith(coll$uri, "s3://cellxgene-census-public-us-west-2/cell-census/"))
  expect_true(coll$exists())
  expect_true(coll$get("census_data")$get("homo_sapiens")$exists())
})

test_that("open_soma stable/default", {
  stable_tiledbsoma_ctx <- new_SOMATileDBContext_for_census(
    get_census_version_description("stable")
  )
  coll_default <- open_soma(tiledbsoma_ctx = stable_tiledbsoma_ctx)
  on.exit(coll_default$close(), add = TRUE)
  coll_stable <- open_soma("stable", tiledbsoma_ctx = stable_tiledbsoma_ctx)
  on.exit(coll_stable$close(), add = TRUE)
  expect_equal(coll_default$uri, coll_stable$uri)
  rm(stable_tiledbsoma_ctx)
})

test_that("open_soma with custom context/config", {
  # create a context with a bogus setting of vfs.s3.region, which should cause
  # open_soma() to fail due to contacting the wrong S3 region.
  ctx <- new_SOMATileDBContext_for_census(
    get_census_version_description("latest"),
    "vfs.s3.region" = "us-east-2"
  )
  expect_error(open_soma(tiledbsoma_ctx = ctx), "Error while listing")
  rm(ctx)
})

test_that("open_soma does not sign AWS S3 requests", {
  Sys.setenv(AWS_ACCESS_KEY_ID = "fake_id", AWS_SECRET_ACCESS_KEY = "fake_key")
  coll <- open_soma_latest_for_test()
  on.exit(coll$close(), add = TRUE)
  expect_true(coll$exists())
  expect_true(coll$get("census_data")$get("homo_sapiens")$exists())
  # Reset the access key and secret - this is necessary because `download_source_h5ad` doesn't support
  # anonymous access and a bogus key will cause the test to fail
  Sys.setenv(AWS_ACCESS_KEY_ID = "", AWS_SECRET_ACCESS_KEY = "")
})

test_that("resolve_census_locator paths", {
  # typical path
  locator <- resolve_census_locator(KNOWN_CENSUS_VERSION, NULL, NULL)
  expect_equal(locator$uri, KNOWN_CENSUS_URI)
  expect_equal(locator$mirror_info$alias, "default")
  expect_equal(locator$mirror_info$provider, "S3")
  expect_equal(locator$mirror_info$region, "us-west-2")

  # non-default mirror
  locator <- resolve_census_locator(KNOWN_CENSUS_VERSION, NULL, "AWS-S3-us-west-2")
  expect_equal(locator$uri, KNOWN_CENSUS_URI)
  expect_false("alias" %in% locator$mirror_info)
  expect_equal(locator$mirror_info$provider, "S3")
  expect_equal(locator$mirror_info$region, "us-west-2")

  # explicit URI: file provider assumed (but that assumption could be overridden
  # by supplying a preconfigured tiledbsoma_ctx to open_soma())
  locator <- resolve_census_locator(KNOWN_CENSUS_VERSION, "file://tmp/soma", NULL)
  expect_equal(locator$uri, "file://tmp/soma")
  expect_null(locator$mirror_info)

  # error paths...
  expect_error(resolve_census_locator("bogus", NULL, NULL), "Census version is not valid")
  expect_error(resolve_census_locator(KNOWN_CENSUS_VERSION, NULL, "bogus"), "Unknown Census mirror")
})
