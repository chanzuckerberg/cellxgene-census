table_iter_is_ok <- function(tbl_iter, stop_after = NULL) {
  if (!is(tbl_iter, "TableReadIter")) {
    return(FALSE)
  }

  n <- 1
  while (!tbl_iter$read_complete()) {
    if (!is.null(stop_after)) {
      if (n > stop_after) {
        break
      }
    }

    tbl <- tbl_iter$read_next()
    if (!is(tbl, "Table")) {
      return(FALSE)
    }
    if (!is(tbl, "ArrowTabular")) {
      return(FALSE)
    }
    if (nrow(tbl) < 1) {
      return(FALSE)
    }

    n <- n + 1
  }

  return(TRUE)
}

# Tests that the object from get_seurat is a non-empty Seurat object
test_seurat <- function(get_seurat_args) {
  this_seurat <- do.call(get_seurat, get_seurat_args)
  expect_true(is(this_seurat, "Seurat"))
  expect_true(ncol(this_seurat) > 0)
}

# Tests that the object is SingleCellExperiment and is a non-empty
test_sce <- function(get_sce_args) {
  this_sce <- do.call(get_single_cell_experiment, get_sce_args)
  expect_true(is(this_sce, "SingleCellExperiment"))
  expect_true(ncol(this_sce) > 0)
}
