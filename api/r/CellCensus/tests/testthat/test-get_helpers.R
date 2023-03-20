test_that("get_experiment", {
  census <- open_soma()

  cases <- list(
    "mus_musculus" = c("Mus musculus", "mus_musculus"),
    "homo_sapiens" = c("Homo sapiens", "homo_sapiens")
  )

  for (org in names(cases)) {
    uri <- census$get("census_data")$get(org)$uri
    for (alias in c(org, cases[[org]])) {
      expect_equal(get_experiment(census, alias)$uri, uri)
    }
  }

  expect_error(get_experiment(census, "bogus"), "Unknown organism")
})

test_that("get_presence_matrix", {
  census <- open_soma()
  datasets <- as.data.frame(census$get("census_info")$get("datasets")$read())
  for (org in c("homo_sapiens", "mus_musculus")) {
    pm <- get_presence_matrix(census, org)
    expect_s4_class(pm, "sparseMatrix")
    expect_equal(nrow(pm), nrow(datasets))
    expect_equal(
      ncol(pm),
      nrow(census$get("census_data")$get(org)$ms$get("RNA")$var$read(column_names = "soma_joinid"))
    )
    expect_equal(min(pm), 0)
    expect_equal(max(pm), 1)
  }
})
