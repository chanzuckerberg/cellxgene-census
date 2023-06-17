table_iter_is_ok <- function(tbl_iter, stop_after = 1) {

if (!is(tbl_iter, "TableReadIter")) return(FALSE)

n <- 1
while (!tbl_iter$read_complete()) {
  
  if(!is.null(stop_after) & n > stop_after) break
  
  tbl <- tbl_iter$read_next()
  if (!is(tbl, "Table")) return(FALSE)
  if (!is(tbl, "ArrowTabular")) return(FALSE)
  if (nrow(tbl) < 1) return(FALSE)
  
  n <- n + 1
}

return(TRUE)
}
