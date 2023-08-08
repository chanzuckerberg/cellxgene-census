# Run as Rscript acceptance-tests-run-script.R
library("cellxgene.census")
library("testthat")
source("./ListReporterToFile.R")

reporter <- ListReporterToFile$new(paste0("acceptance-tests-logs-", Sys.Date(), ".csv"))
test_file("acceptance-tests.R", reporter = reporter)
