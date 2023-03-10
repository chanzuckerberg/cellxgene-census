test_that("get_source_h5ad_uri", {
  census <- open_soma()
  datasets <- as.data.frame(census$get("census_info")$get("datasets")$read(
    column_names = c("dataset_id", "dataset_h5ad_path")
  ))
  datasets <- datasets[sample(nrow(datasets), 10), ]

  apply(datasets, 1, function(dataset) {
    dataset <- as.list(dataset)

    loc <- get_source_h5ad_uri(dataset$dataset_id)

    expect_true(endsWith(loc$uri, paste("/", dataset$dataset_h5ad_path, sep = "")))
    expect_equal(loc$s3_region, unname(tiledb::config(census$ctx)["vfs.s3.region"]))
    # check URI joining with trailing slashes
    expect_false(endsWith(loc$uri, paste("//", dataset$dataset_h5ad_path, sep = "")))
  })

  expect_error(get_source_h5ad_uri("bogus"))
})
