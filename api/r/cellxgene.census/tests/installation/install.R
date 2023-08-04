options(error = function() {
  q("no", status = 1, runLast = FALSE)
})
Sys.setenv(MAKE = "make -j8")

#######################################################################################

install.packages(
  "tiledb",
  version = "0.20.2",
  repos = c("https://tiledb-inc.r-universe.dev", "https://cloud.r-project.org")
)

install.packages(
  "cellxgene.census",
  repos = c("https://chanzuckerberg.r-universe.dev", "https://cloud.r-project.org")
)

tiledbsoma::show_package_versions()
print(as.data.frame(
  cellxgene.census::open_soma()$get("census_info")$get("summary")$read()$concat()
))
