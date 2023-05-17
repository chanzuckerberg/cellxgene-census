test_that("get_source_h5ad_uri", {
  census_region <- get_census_version_description("latest")$soma.s3_region
  census <- open_soma("latest")
  on.exit(census$close(), add = TRUE)
  datasets <- as.data.frame(census$get("census_info")$get("datasets")$read(
    column_names = c("dataset_id", "dataset_h5ad_path")
  ))
  datasets <- datasets[sample(nrow(datasets), 10), ]

  apply(datasets, 1, function(dataset) {
    dataset <- as.list(dataset)

    loc <- get_source_h5ad_uri(dataset$dataset_id)

    expect_true(endsWith(loc$uri, paste("/", dataset$dataset_h5ad_path, sep = "")))
    expect_equal(loc$s3_region, census_region)
    # check URI joining with trailing slashes
    expect_false(endsWith(loc$uri, paste("//", dataset$dataset_h5ad_path, sep = "")))
  })

  expect_error(get_source_h5ad_uri("bogus"))
})

test_that("download_source_h5ad", {
  # find the ~smallest dataset
  census <- open_soma()
  on.exit(census$close(), add = TRUE)
  datasets <- as.data.frame(census$get("census_info")$get("datasets")$read(
    column_names = c("dataset_id", "dataset_total_cell_count")
  ))
  dataset <- as.list(datasets[which.min(datasets$dataset_total_cell_count), ])

  # fetch its h5ad
  fn <- tempfile("data_", fileext = ".h5ad")
  withr::defer(unlink(fn))
  expect_false(file.exists(fn))

  download_source_h5ad(dataset$dataset_id, fn)
  expect_true(file.exists(fn))
  expect_gt(file.size(fn), 0)

  # refuse overwrite
  expect_error(download_source_h5ad(dataset$dataset_id, fn))

  # refuse directory
  expect_error(download_source_h5ad(dataset$dataset_id, tempdir()))
})
