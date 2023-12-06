# Children class of testthat::ListReporter
# writes test results to a file as it runs them

ListReporterToFile <- R6::R6Class("ListReporterToFile",
  inherit = ListReporter,
  public = list(
    initialize = function(to_file = "acceptance-tests-logs.csv") {
      private$to_file <- to_file

      file_handle <- file(private$to_file)
      base::writeLines("test,user,system,real,test_result", con = file_handle)
      close(file_handle)

      super$initialize()
    },
    start_test = function(context, test) {
      # print stdout for live progress
      cat("START TEST (", as.character(Sys.time()), "): ", test, "\n")
      super$start_test(context, test)
    },
    end_test = function(context, test) {
      super$end_test(context, test)
      this_result <- super$get_results()

      # get last result
      this_result <- this_result[[length(this_result)]]

      # write to file
      file_handle <- file(private$to_file, "a")
      base::writeLines(
        con = file_handle,
        text = paste(this_result$test,
          this_result$user,
          this_result$system,
          this_result$real,
          private$convert_test_output_to_string(this_result$results),
          sep = ","
        )
      )
      close(file_handle)

      # print stdout for live progress
      cat("END TEST (", as.character(Sys.time()), "): ", test, "\n")
    }
  ),
  private = list(
    to_file = NULL,
    convert_test_output_to_string = function(results) {
      results_list <- lapply(results, function(x) {
        call_title <- x$srcref
        call_result <- gsub("\n+", " ", as.character(x))
        paste0(call_title, ": ", call_result)
      })
      return(paste0(as.vector(results_list), collapse = "; "))
    }
  )
)
