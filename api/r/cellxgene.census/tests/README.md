# Test README

This directory contains tests of the `cellxgene.census` R package API, _and_ the use of the API on the live "corpus", i.e., data in the public Census S3 bucket. The tests use the R package `tessthat`.

In addition, a set of acceptance (expensive) tests are available and `testthat` does not run them by default (see [section below](#acceptance-expensive-tests)).

Tests can be run in the usual manner. First, ensure you have `cellxgene-census` and `testthat` installed, e.g., from the top-level repo directory:

Then run the tests from R in the repo root folder:

```r
library("testthat")
library("cellxgene.census")
test_dir("./api/r/cellxgene.census/tests/")
```

## Acceptance (expensive) tests

These tests are periodically run, and are not part of CI due to their overhead.

These tests use a modified `Reporter` from `testthat` to record running times of each test in a `csv` file. To run the test execute the following command:

```shell
Rscript ./api/r/cellxgene.census/tests/testthat/acceptance-tests-run-script.R > stdout.txt
```

This command will result in two files:

- `stdout.txt` with the test progress logs.
- `acceptance-tests-logs-[YYY]-[MM]-[DD].csv` with the running times and test outputs.

When run, please record the results in this file (below) and commit the change to git. Please include the following information:

- date
- config:
  - EC2 instance type and any system config (i.e., swap)
  - host and OS as reported by `uname -a`
  - R session info `library("cellxgene.census"); sessionInfo()`
  - The Census version used for the test (i.e., the version aliased as `latest`). This can be easily captured using `cellxgene.census::get_census_version_description('latest')`
- any run notes
- full output of:
  - `stdout.txt`
  - `acceptance-tests-logs-[YYY]-[MM]-[DD].csv`

### 2023-10-23

- Host: EC2 instance type: `r6id.x32xlarge`, all nvme mounted as swap.
- Uname: Linux 6.2.0-1015-aws #15~22.04.1-Ubuntu SMP Fri Oct  6 21:37:24 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Census version

```r
> cellxgene.census::get_census_version_description('2023-10-23')
$release_date
[1] ""

$release_build
[1] "2023-10-23"

$soma.uri
[1] "s3://cellxgene-data-public/cell-census/2023-10-23/soma/"

$soma.relative_uri
[1] "/cell-census/2023-10-23/soma/"

$soma.s3_region
[1] "us-west-2"

$h5ads.uri
[1] "s3://cellxgene-data-public/cell-census/2023-10-23/h5ads/"

$h5ads.relative_uri
[1] "/cell-census/2023-10-23/h5ads/"

$h5ads.s3_region
[1] "us-west-2"

$do_not_delete
[1] TRUE

$lts
[1] FALSE

$alias
[1] ""

$census_version
[1] "2023-10-23"
```

- R session info

```r
> library("cellxgene.census"); sessionInfo()
R version 4.3.2 (2023-10-31)
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Ubuntu 22.04.3 LTS

Matrix products: default
BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0
LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.10.0

locale:
 [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8
 [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8
 [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C
[10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C

time zone: Etc/UTC
tzcode source: system (glibc)

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base

other attached packages:
[1] cellxgene.census_1.7.0

loaded via a namespace (and not attached):
 [1] Matrix_1.6-2        bit_4.0.5           jsonlite_1.8.7
 [4] dplyr_1.1.3         compiler_4.3.2      tidyselect_1.2.0
 [7] Rcpp_1.0.11         xml2_1.3.5          urltools_1.7.3
[10] assertthat_0.2.1    tiledbsoma_1.5.1    arrow_13.0.0.1
[13] triebeard_0.4.1     lattice_0.22-5      R6_2.5.1
[16] generics_0.1.3      curl_5.1.0          RcppSpdlog_0.0.14
[19] tibble_3.2.1        aws.s3_0.3.21       pillar_1.9.0
[22] rlang_1.1.2         utf8_1.2.4          fs_1.6.3
[25] bit64_4.0.5         cli_3.6.1           magrittr_2.0.3
[28] spdl_0.0.5          digest_0.6.33       grid_4.3.2
[31] base64enc_0.1-3     lifecycle_1.0.4     vctrs_0.6.4
[34] glue_1.6.2          data.table_1.14.8   aws.signature_0.6.0
[37] fansi_1.0.5         purrr_1.0.2         httr_1.4.7
[40] tools_4.3.2         pkgconfig_2.0.3
```

- `stdout.txt`

```text
START TEST ( 2023-11-14 22:08:16.174153 ):  test_load_obs_human
END TEST ( 2023-11-14 22:08:22.98588 ):  test_load_obs_human
START TEST ( 2023-11-14 22:08:22.98783 ):  test_load_var_human
END TEST ( 2023-11-14 22:08:25.442495 ):  test_load_var_human
START TEST ( 2023-11-14 22:08:25.444479 ):  test_load_obs_mouse
END TEST ( 2023-11-14 22:08:28.024443 ):  test_load_obs_mouse
START TEST ( 2023-11-14 22:08:28.026742 ):  test_load_var_mouse
END TEST ( 2023-11-14 22:08:30.551841 ):  test_load_var_mouse
START TEST ( 2023-11-14 22:08:30.553875 ):  test_incremental_read_obs_human
END TEST ( 2023-11-14 22:08:34.939039 ):  test_incremental_read_obs_human
START TEST ( 2023-11-14 22:08:34.941016 ):  test_incremental_read_var_human
END TEST ( 2023-11-14 22:08:37.237972 ):  test_incremental_read_var_human
START TEST ( 2023-11-14 22:08:37.240094 ):  test_incremental_read_obs_mouse
END TEST ( 2023-11-14 22:08:39.94039 ):  test_incremental_read_obs_mouse
START TEST ( 2023-11-14 22:08:39.942593 ):  test_incremental_read_var_mouse
END TEST ( 2023-11-14 22:08:42.370465 ):  test_incremental_read_var_mouse
START TEST ( 2023-11-14 22:08:42.372389 ):  test_incremental_read_X_human
END TEST ( 2023-11-14 22:34:22.488574 ):  test_incremental_read_X_human
START TEST ( 2023-11-14 22:34:22.492159 ):  test_incremental_read_X_human-large-buffer-size
END TEST ( 2023-11-14 23:09:18.232126 ):  test_incremental_read_X_human-large-buffer-size
START TEST ( 2023-11-14 23:09:18.264509 ):  test_incremental_read_X_mouse
END TEST ( 2023-11-14 23:11:56.351876 ):  test_incremental_read_X_mouse
START TEST ( 2023-11-14 23:11:56.353827 ):  test_incremental_read_X_mouse-large-buffer-size
END TEST ( 2023-11-14 23:14:17.630646 ):  test_incremental_read_X_mouse-large-buffer-size
START TEST ( 2023-11-14 23:14:17.632895 ):  test_incremental_query_human_brain
END TEST ( 2023-11-14 23:15:55.335929 ):  test_incremental_query_human_brain
START TEST ( 2023-11-14 23:15:55.338135 ):  test_incremental_query_human_aorta
END TEST ( 2023-11-14 23:16:08.916269 ):  test_incremental_query_human_aorta
START TEST ( 2023-11-14 23:16:08.918454 ):  test_incremental_query_mouse_brain
END TEST ( 2023-11-14 23:16:37.376596 ):  test_incremental_query_mouse_brain
START TEST ( 2023-11-14 23:16:37.379342 ):  test_incremental_query_mouse_aorta
END TEST ( 2023-11-14 23:16:47.94446 ):  test_incremental_query_mouse_aorta
START TEST ( 2023-11-14 23:16:47.947125 ):  test_seurat_small-query
END TEST ( 2023-11-14 23:17:14.07548 ):  test_seurat_small-query
START TEST ( 2023-11-14 23:17:14.077846 ):  test_seurat_10K-cells-human
END TEST ( 2023-11-14 23:17:38.149094 ):  test_seurat_10K-cells-human
START TEST ( 2023-11-14 23:17:38.15124 ):  test_seurat_100K-cells-human
END TEST ( 2023-11-14 23:19:20.828073 ):  test_seurat_100K-cells-human
START TEST ( 2023-11-14 23:19:20.830142 ):  test_seurat_250K-cells-human
END TEST ( 2023-11-14 23:23:40.847132 ):  test_seurat_250K-cells-human
START TEST ( 2023-11-14 23:23:40.84913 ):  test_seurat_500K-cells-human
END TEST ( 2023-11-14 23:25:15.641259 ):  test_seurat_500K-cells-human
START TEST ( 2023-11-14 23:25:15.643847 ):  test_seurat_750K-cells-human
END TEST ( 2023-11-14 23:27:27.224292 ):  test_seurat_750K-cells-human
START TEST ( 2023-11-14 23:27:27.226668 ):  test_seurat_1M-cells-human
END TEST ( 2023-11-14 23:30:22.724453 ):  test_seurat_1M-cells-human
START TEST ( 2023-11-14 23:30:22.726397 ):  test_seurat_common-tissue
END TEST ( 2023-11-14 23:36:58.342319 ):  test_seurat_common-tissue
START TEST ( 2023-11-14 23:36:58.344535 ):  test_seurat_common-tissue-large-buffer-size
END TEST ( 2023-11-14 23:43:24.081703 ):  test_seurat_common-tissue-large-buffer-size
START TEST ( 2023-11-14 23:43:24.084217 ):  test_seurat_common-cell-type
END TEST ( 2023-11-15 00:20:01.464067 ):  test_seurat_common-cell-type
START TEST ( 2023-11-15 00:20:01.467048 ):  test_seurat_common-cell-type-large-buffer-size
END TEST ( 2023-11-15 01:22:22.319294 ):  test_seurat_common-cell-type-large-buffer-size
START TEST ( 2023-11-15 01:22:22.324062 ):  test_seurat_whole-enchilada-large-buffer-size
END TEST ( 2023-11-15 01:22:22.331545 ):  test_seurat_whole-enchilada-large-buffer-size
START TEST ( 2023-11-15 01:22:22.333495 ):  test_sce_small-query
END TEST ( 2023-11-15 01:22:48.192984 ):  test_sce_small-query
START TEST ( 2023-11-15 01:22:48.195162 ):  test_sce_10K-cells-human
END TEST ( 2023-11-15 01:23:07.706087 ):  test_sce_10K-cells-human
START TEST ( 2023-11-15 01:23:07.708716 ):  test_sce_100K-cells-human
END TEST ( 2023-11-15 01:24:11.72169 ):  test_sce_100K-cells-human
START TEST ( 2023-11-15 01:24:11.724583 ):  test_sce_250K-cells-human
END TEST ( 2023-11-15 01:26:35.928612 ):  test_sce_250K-cells-human
START TEST ( 2023-11-15 01:26:35.933126 ):  test_sce_500K-cells-human
END TEST ( 2023-11-15 01:32:20.991126 ):  test_sce_500K-cells-human
START TEST ( 2023-11-15 01:32:20.994043 ):  test_sce_750K-cells-human
END TEST ( 2023-11-15 01:39:21.034239 ):  test_sce_750K-cells-human
START TEST ( 2023-11-15 01:39:21.036942 ):  test_sce_1M-cells-human
END TEST ( 2023-11-15 01:48:36.249403 ):  test_sce_1M-cells-human
START TEST ( 2023-11-15 01:48:36.252596 ):  test_sce_common-tissue
END TEST ( 2023-11-15 01:51:28.563483 ):  test_sce_common-tissue
START TEST ( 2023-11-15 01:51:28.567205 ):  test_sce_common-tissue-large-buffer-size
END TEST ( 2023-11-15 01:54:16.310687 ):  test_sce_common-tissue-large-buffer-size
START TEST ( 2023-11-15 01:54:16.314651 ):  test_sce_common-cell-type
END TEST ( 2023-11-15 04:09:16.883595 ):  test_sce_common-cell-type
START TEST ( 2023-11-15 04:09:16.888096 ):  test_sce_common-cell-type-large-buffer-size
END TEST ( 2023-11-15 06:33:44.356152 ):  test_sce_common-cell-type-large-buffer-size
START TEST ( 2023-11-15 06:33:44.367543 ):  test_sce_whole-enchilada-large-buffer-size
END TEST ( 2023-11-15 06:33:44.40467 ):  test_sce_whole-enchilada-large-buffer-size
```

- `acceptance-tests-logs-2023-11-14.csv`

```text
test,user,system,real,test_result
test_load_obs_human,10.872,45.086,6.811,expect_true(nrow(obs_df) > 0): expectation_success: nrow(obs_df) > 0 is not TRUE
test_load_var_human,0.426,0.638999999999996,2.448,expect_true(nrow(var_df) > 0): expectation_success: nrow(var_df) > 0 is not TRUE
test_load_obs_mouse,1.673,6.315,2.579,expect_true(nrow(obs_df) > 0): expectation_success: nrow(obs_df) > 0 is not TRUE
test_load_var_mouse,0.51,1.719,2.524,expect_true(nrow(var_df) > 0): expectation_success: nrow(var_df) > 0 is not TRUE
test_incremental_read_obs_human,10.135,35.531,4.384,expect_true(table_iter_is_ok(obs_iter)): expectation_success: table_iter_is_ok(obs_iter) is not TRUE
test_incremental_read_var_human,0.457000000000001,0.575000000000003,2.296,expect_true(table_iter_is_ok(var_iter)): expectation_success: table_iter_is_ok(var_iter) is not TRUE
test_incremental_read_obs_mouse,1.701,7.48899999999999,2.699,expect_true(table_iter_is_ok(obs_iter)): expectation_success: table_iter_is_ok(obs_iter) is not TRUE
test_incremental_read_var_mouse,0.701999999999998,1.917,2.427,expect_true(table_iter_is_ok(var_iter)): expectation_success: table_iter_is_ok(var_iter) is not TRUE
test_incremental_read_X_human,9478.638,20812.554,1540.115,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_human-large-buffer-size,9515.992,58285.962,2095.726,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_mouse,1003.918,1709.78600000001,158.077,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_mouse-large-buffer-size,1003.373,1704.70299999999,141.275,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_query_human_brain,222.373,217.176999999996,97.7020000000002,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_incremental_query_human_aorta,11.768,25.8989999999903,13.5769999999998,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_incremental_query_mouse_brain,69.893,62.8429999999935,28.4569999999999,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_incremental_query_mouse_aorta,21.607,15.0279999999912,10.5630000000001,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_seurat_small-query,22.7649999999994,24.9000000000087,26.1269999999995,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_10K-cells-human,18.5760000000009,14.9829999999929,24.0699999999997,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_100K-cells-human,100.204999999998,81.8509999999951,102.676,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_250K-cells-human,260.911,210.860000000001,260.016000000001,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_500K-cells-human,188.810000000001,300.986000000004,94.7910000000002,test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_750K-cells-human,286.100999999999,476.747999999992,131.579,test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_1M-cells-human,395.574000000001,664.184999999998,175.497,test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_common-tissue,479.936999999998,267.138000000006,395.615000000001,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_common-tissue-large-buffer-size,475.071,256.587,385.736,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_common-cell-type,3491.464,25131.535,2197.378,test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_common-cell-type-large-buffer-size,3723.685,20053.295,3740.85,test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_whole-enchilada-large-buffer-size,0.00599999999758438,0,0.0069999999996071,expect_true(TRUE): expectation_success: TRUE is not TRUE
test_sce_small-query,20.3489999999983,32.1600000000035,25.8549999999996,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_10K-cells-human,14.1660000000011,15.6140000000014,19.509,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_100K-cells-human,73.7009999999973,71.1290000000008,64.0109999999986,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_250K-cells-human,169.846000000001,211.741999999998,144.201000000001,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_500K-cells-human,327.071,422.800999999992,345.055999999999,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_750K-cells-human,510.853000000003,556.207999999984,420.038,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_1M-cells-human,681.242000000002,793.367999999988,555.210999999999,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_common-tissue,302.630999999998,240.524999999994,172.307999999999,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_common-tissue-large-buffer-size,298.978000000003,251.656999999977,167.74,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_common-cell-type,6661.194,14059.888,8100.566,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_common-cell-type-large-buffer-size,6539.15100000001,12091.293,8667.461,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_whole-enchilada-large-buffer-size,0.010999999998603,0.00699999998323619,0.0349999999998545,expect_true(TRUE): expectation_success: TRUE is not TRUE
```

### 2023-07-15

- Host: EC2 instance type: `r6id.x32xlarge`, all nvme mounted as swap.
- Uname: Linux ip-172-31-62-52 5.19.0-1028-aws #29~22.04.1-Ubuntu SMP Tue Jun 20 19:12:11 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Census version

```r
> cellxgene.census::get_census_version_description('latest')
$release_date
[1] ""

$release_build
[1] "2023-07-10"

$soma.uri
[1] "s3://cellxgene-data-public/cell-census/2023-07-10/soma/"

$soma.s3_region
[1] "us-west-2"

$h5ads.uri
[1] "s3://cellxgene-data-public/cell-census/2023-07-10/h5ads/"

$h5ads.s3_region
[1] "us-west-2"

$alias
[1] "latest"

$census_version
[1] "latest"
```

- R session info

```r
> library("cellxgene.census"); sessionInfo()
R version 4.3.0 (2023-04-21)
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Ubuntu 22.04.2 LTS

Matrix products: default
BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0
LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.10.0

locale:
 [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8
 [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8
 [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C
[10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C

time zone: America/Los_Angeles
tzcode source: system (glibc)

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base

other attached packages:
[1] cellxgene.census_0.0.0.9000

loaded via a namespace (and not attached):
 [1] vctrs_0.6.3           httr_1.4.6            cli_3.6.1
 [4] tiledbsoma_0.0.0.9031 rlang_1.1.1           purrr_1.0.1
 [7] assertthat_0.2.1      data.table_1.14.8     jsonlite_1.8.5
[10] glue_1.6.2            bit_4.0.5             triebeard_0.4.1
[13] grid_4.3.0            RcppSpdlog_0.0.13     base64enc_0.1-3
[16] lifecycle_1.0.3       compiler_4.3.0        fs_1.6.2
[19] Rcpp_1.0.10           aws.s3_0.3.21         lattice_0.21-8
[22] digest_0.6.31         R6_2.5.1              tidyselect_1.2.0
[25] curl_5.0.1            magrittr_2.0.3        urltools_1.7.3
[28] Matrix_1.5-4.1        tools_4.3.0           bit64_4.0.5
[31] aws.signature_0.6.0   spdl_0.0.5            arrow_12.0.1
[34] xml2_1.3.4
```

- `stdout.txt`

```text
START TEST ( 2023-07-14 11:53:28.178227 ):  test_load_obs_human
END TEST ( 2023-07-14 11:53:35.399914 ):  test_load_obs_human
START TEST ( 2023-07-14 11:53:35.40381 ):  test_load_var_human
END TEST ( 2023-07-14 11:53:37.241811 ):  test_load_var_human
START TEST ( 2023-07-14 11:53:37.243805 ):  test_load_obs_mouse
END TEST ( 2023-07-14 11:53:39.325767 ):  test_load_obs_mouse
START TEST ( 2023-07-14 11:53:39.327926 ):  test_load_var_mouse
END TEST ( 2023-07-14 11:53:40.895862 ):  test_load_var_mouse
START TEST ( 2023-07-14 11:53:40.899131 ):  test_incremental_read_obs_human
END TEST ( 2023-07-14 11:53:46.524585 ):  test_incremental_read_obs_human
START TEST ( 2023-07-14 11:53:46.526778 ):  test_incremental_read_var_human
END TEST ( 2023-07-14 11:53:47.874486 ):  test_incremental_read_var_human
START TEST ( 2023-07-14 11:53:47.877107 ):  test_incremental_read_obs_mouse
END TEST ( 2023-07-14 11:53:50.236715 ):  test_incremental_read_obs_mouse
START TEST ( 2023-07-14 11:53:50.239098 ):  test_incremental_read_var_mouse
END TEST ( 2023-07-14 11:53:51.736903 ):  test_incremental_read_var_mouse
START TEST ( 2023-07-14 11:53:51.739117 ):  test_incremental_read_X_human
END TEST ( 2023-07-14 12:14:34.869316 ):  test_incremental_read_X_human
START TEST ( 2023-07-14 12:14:34.871589 ):  test_incremental_read_X_human-large-buffer-size
END TEST ( 2023-07-14 12:49:30.484771 ):  test_incremental_read_X_human-large-buffer-size
START TEST ( 2023-07-14 12:49:30.570718 ):  test_incremental_read_X_mouse
END TEST ( 2023-07-14 12:54:44.455472 ):  test_incremental_read_X_mouse
START TEST ( 2023-07-14 12:54:44.466457 ):  test_incremental_read_X_mouse-large-buffer-size
END TEST ( 2023-07-14 12:56:15.48859 ):  test_incremental_read_X_mouse-large-buffer-size
START TEST ( 2023-07-14 12:56:15.491021 ):  test_incremental_query_human_brain
END TEST ( 2023-07-14 12:57:17.430526 ):  test_incremental_query_human_brain
START TEST ( 2023-07-14 12:57:17.434666 ):  test_incremental_query_human_aorta
END TEST ( 2023-07-14 12:57:29.836529 ):  test_incremental_query_human_aorta
START TEST ( 2023-07-14 12:57:29.838435 ):  test_incremental_query_mouse_brain
END TEST ( 2023-07-14 12:57:41.417177 ):  test_incremental_query_mouse_brain
START TEST ( 2023-07-14 12:57:41.419112 ):  test_incremental_query_mouse_aorta
END TEST ( 2023-07-14 12:57:48.16363 ):  test_incremental_query_mouse_aorta
START TEST ( 2023-07-14 12:57:48.166098 ):  test_seurat_small-query
END TEST ( 2023-07-14 12:58:09.744611 ):  test_seurat_small-query
START TEST ( 2023-07-14 12:58:09.746436 ):  test_seurat_10K-cells-human
END TEST ( 2023-07-14 12:58:19.528256 ):  test_seurat_10K-cells-human
START TEST ( 2023-07-14 12:58:19.530055 ):  test_seurat_100K-cells-human
END TEST ( 2023-07-14 12:58:52.22588 ):  test_seurat_100K-cells-human
START TEST ( 2023-07-14 12:58:52.227741 ):  test_seurat_250K-cells-human
END TEST ( 2023-07-14 13:00:12.885999 ):  test_seurat_250K-cells-human
START TEST ( 2023-07-14 13:00:12.887802 ):  test_seurat_500K-cells-human
END TEST ( 2023-07-14 13:03:13.64018 ):  test_seurat_500K-cells-human
START TEST ( 2023-07-14 13:03:13.641989 ):  test_seurat_750K-cells-human
END TEST ( 2023-07-14 13:08:09.243155 ):  test_seurat_750K-cells-human
START TEST ( 2023-07-14 13:08:09.800513 ):  test_seurat_1M-cells-human
END TEST ( 2023-07-14 13:14:21.120332 ):  test_seurat_1M-cells-human
START TEST ( 2023-07-14 13:14:21.122229 ):  test_seurat_common-tissue
END TEST ( 2023-07-14 13:18:23.092255 ):  test_seurat_common-tissue
START TEST ( 2023-07-14 13:18:23.094857 ):  test_seurat_common-tissue-large-buffer-size
END TEST ( 2023-07-14 13:22:31.049091 ):  test_seurat_common-tissue-large-buffer-size
START TEST ( 2023-07-14 13:22:31.050861 ):  test_seurat_common-cell-type
END TEST ( 2023-07-14 13:39:16.80712 ):  test_seurat_common-cell-type
START TEST ( 2023-07-14 13:39:16.818948 ):  test_seurat_common-cell-type-large-buffer-size
END TEST ( 2023-07-14 15:06:24.936444 ):  test_seurat_common-cell-type-large-buffer-size
START TEST ( 2023-07-14 15:06:24.943206 ):  test_seurat_whole-enchilada-large-buffer-size
END TEST ( 2023-07-14 15:06:24.955094 ):  test_seurat_whole-enchilada-large-buffer-size
START TEST ( 2023-07-14 15:06:24.958625 ):  test_sce_small-query
END TEST ( 2023-07-14 15:06:53.449669 ):  test_sce_small-query
START TEST ( 2023-07-14 15:06:53.451782 ):  test_sce_10K-cells-human
END TEST ( 2023-07-14 15:07:03.753756 ):  test_sce_10K-cells-human
START TEST ( 2023-07-14 15:07:03.756331 ):  test_sce_100K-cells-human
END TEST ( 2023-07-14 15:07:39.928058 ):  test_sce_100K-cells-human
START TEST ( 2023-07-14 15:07:39.931532 ):  test_sce_250K-cells-human
END TEST ( 2023-07-14 15:08:59.480538 ):  test_sce_250K-cells-human
START TEST ( 2023-07-14 15:08:59.482945 ):  test_sce_500K-cells-human
END TEST ( 2023-07-14 15:12:02.190109 ):  test_sce_500K-cells-human
START TEST ( 2023-07-14 15:12:02.192345 ):  test_sce_750K-cells-human
END TEST ( 2023-07-14 15:17:29.745159 ):  test_sce_750K-cells-human
START TEST ( 2023-07-14 15:17:29.748402 ):  test_sce_1M-cells-human
END TEST ( 2023-07-14 15:22:46.696071 ):  test_sce_1M-cells-human
START TEST ( 2023-07-14 15:22:46.69859 ):  test_sce_common-tissue
END TEST ( 2023-07-14 15:25:59.307055 ):  test_sce_common-tissue
START TEST ( 2023-07-14 15:25:59.309585 ):  test_sce_common-tissue-large-buffer-size
END TEST ( 2023-07-14 15:29:05.180097 ):  test_sce_common-tissue-large-buffer-size
START TEST ( 2023-07-14 15:29:05.182871 ):  test_sce_common-cell-type
END TEST ( 2023-07-14 17:16:40.55286 ):  test_sce_common-cell-type
START TEST ( 2023-07-14 17:16:40.557382 ):  test_sce_common-cell-type-large-buffer-size
END TEST ( 2023-07-14 19:37:35.293807 ):  test_sce_common-cell-type-large-buffer-size
START TEST ( 2023-07-14 19:37:35.299182 ):  test_sce_whole-enchilada-large-buffer-size
END TEST ( 2023-07-14 19:37:35.305957 ):  test_sce_whole-enchilada-large-buffer-size
```

- `acceptance-tests-logs-2023-07-14.csv`

```text
test,user,system,real,test_result
test_load_obs_human,15.906,89.247,7.221,expect_true(nrow(obs_df) > 0): expectation_success: nrow(obs_df) > 0 is not TRUE
test_load_var_human,0.494,0.52300000000001,1.831,expect_true(nrow(var_df) > 0): expectation_success: nrow(var_df) > 0 is not TRUE
test_load_obs_mouse,2.221,6.60799999999999,2.081,expect_true(nrow(obs_df) > 0): expectation_success: nrow(obs_df) > 0 is not TRUE
test_load_var_mouse,0.385999999999999,0.501999999999995,1.567,expect_true(nrow(var_df) > 0): expectation_success: nrow(var_df) > 0 is not TRUE
test_incremental_read_obs_human,15.758,102.911,5.624,expect_true(table_iter_is_ok(obs_iter)): expectation_success: table_iter_is_ok(obs_iter) is not TRUE
test_incremental_read_var_human,0.390999999999998,0.533999999999992,1.346,expect_true(table_iter_is_ok(var_iter)): expectation_success: table_iter_is_ok(var_iter) is not TRUE
test_incremental_read_obs_mouse,2.95200000000001,11.719,2.359,expect_true(table_iter_is_ok(obs_iter)): expectation_success: table_iter_is_ok(obs_iter) is not TRUE
test_incremental_read_var_mouse,0.412999999999997,0.378999999999991,1.497,expect_true(table_iter_is_ok(var_iter)): expectation_success: table_iter_is_ok(var_iter) is not TRUE
test_incremental_read_X_human,8126.569,12080.887,1243.129,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_human-large-buffer-size,8129.084,83266.503,2095.526,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_mouse,950.918,1498.723,313.867,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_mouse-large-buffer-size,941.432000000001,1447.306,91.02,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_query_human_brain,202.976999999999,245.77900000001,61.9389999999999,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_incremental_query_human_aorta,15.2920000000013,87.8139999999985,12.4009999999998,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_incremental_query_mouse_brain,46.5919999999969,59.3540000000066,11.578,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_incremental_query_mouse_aorta,14.3100000000013,10.6429999999964,6.74299999999994,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_seurat_small-query,23.9830000000002,82.7459999999992,21.578,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_10K-cells-human,5.35099999999875,4.23699999999371,9.78099999999995,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_100K-cells-human,32.75,29.4389999999985,32.6950000000002,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_250K-cells-human,88.8899999999994,73.1219999999885,80.6570000000002,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_500K-cells-human,203.519,153.688000000009,180.751,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_750K-cells-human,319.052,240.021999999997,295.6,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_1M-cells-human,399.764000000003,306.130999999994,371.32,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_common-tissue,362.931999999997,274.218999999997,241.969,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_common-tissue-large-buffer-size,359.698,260.566000000006,247.953,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_common-cell-type,3382.376,16645.538,1005.753,test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_common-cell-type-large-buffer-size,3376.691,49904.262,5228.115,test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_whole-enchilada-large-buffer-size,0.00799999999799184,0.00099999998928979,0.0100000000002183,expect_true(TRUE): expectation_success: TRUE is not TRUE
test_sce_small-query,28.0150000000031,102.805999999982,28.4889999999996,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_10K-cells-human,5.32399999999689,5.06299999999464,10.2999999999993,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_100K-cells-human,30.0080000000016,540.113000000012,36.1700000000001,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_250K-cells-human,84.3679999999986,77.1530000000203,79.5490000000009,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_500K-cells-human,195.394,1590.212,182.706,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_750K-cells-human,278.846000000001,312.385000000009,327.552,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_1M-cells-human,369.940000000002,287.795000000013,316.947,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_common-tissue,334.884999999998,266.495999999985,192.607,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_common-tissue-large-buffer-size,333.330000000002,239.800999999978,185.868,test_sce(test_args): expectation_success: is(this_sce, "SingleCellExperiment") is not TRUE ; test_sce(test_args): expectation_success: ncol(this_sce) > 0 is not TRUE
test_sce_common-cell-type,5363.865,41800.937,6455.368,test_sce(test_args): Error: Error in `asMethod(object)`: unable to coerce from TsparseMatrix to [CR]sparseMatrixwhen length of 'i' slot exceeds 2^31-1
test_sce_common-cell-type-large-buffer-size,5398.502,89696.129,8454.734,test_sce(test_args): Error: Error in `asMethod(object)`: unable to coerce from TsparseMatrix to [CR]sparseMatrixwhen length of 'i' slot exceeds 2^31-1
test_sce_whole-enchilada-large-buffer-size,0.00600000000122236,0,0.00599999999758438,expect_true(TRUE): expectation_success: TRUE is not TRUE
```

### 2023-07-02

- Host: EC2 instance type: `r6id.x32xlarge`, all nvme mounted as swap.
- Uname: Linux ip-172-31-62-52 5.19.0-1028-aws #29~22.04.1-Ubuntu SMP Tue Jun 20 19:12:11 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Census version

```r
> cellxgene.census::get_census_version_description('latest')
$release_date
[1] ""

$release_build
[1] "2023-06-28"

$soma.uri
[1] "s3://cellxgene-data-public/cell-census/2023-06-28/soma/"

$soma.s3_region
[1] "us-west-2"

$h5ads.uri
[1] "s3://cellxgene-data-public/cell-census/2023-06-28/h5ads/"

$h5ads.s3_region
[1] "us-west-2"

$alias
[1] "latest"

$census_version
[1] "latest"
```

- R session info

```r
> library("cellxgene.census"); sessionInfo()
R version 4.3.0 (2023-04-21)
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Ubuntu 22.04.2 LTS

Matrix products: default
BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0
LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.10.0

locale:
 [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8
 [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8
 [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C
[10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C

time zone: America/Los_Angeles
tzcode source: system (glibc)

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base

other attached packages:
[1] cellxgene.census_0.0.0.9000

loaded via a namespace (and not attached):
 [1] vctrs_0.6.3           httr_1.4.6            cli_3.6.1
 [4] tiledbsoma_0.0.0.9030 rlang_1.1.1           purrr_1.0.1
 [7] assertthat_0.2.1      data.table_1.14.8     jsonlite_1.8.5
[10] glue_1.6.2            bit_4.0.5             triebeard_0.4.1
[13] grid_4.3.0            RcppSpdlog_0.0.13     base64enc_0.1-3
[16] lifecycle_1.0.3       compiler_4.3.0        fs_1.6.2
[19] Rcpp_1.0.10           aws.s3_0.3.21         lattice_0.21-8
[22] digest_0.6.31         R6_2.5.1              tidyselect_1.2.0
[25] curl_5.0.1            magrittr_2.0.3        urltools_1.7.3
[28] Matrix_1.5-4.1        tools_4.3.0           bit64_4.0.5
[31] aws.signature_0.6.0   spdl_0.0.5            arrow_12.0.1
[34] xml2_1.3.4

```

- `stdout.txt`

```text
START TEST ( 2023-07-02 14:46:28.791692 ):  test_load_obs_human
END TEST ( 2023-07-02 14:46:35.845693 ):  test_load_obs_human
START TEST ( 2023-07-02 14:46:35.847513 ):  test_load_var_human
END TEST ( 2023-07-02 14:46:37.550566 ):  test_load_var_human
START TEST ( 2023-07-02 14:46:37.552496 ):  test_load_obs_mouse
END TEST ( 2023-07-02 14:46:39.540367 ):  test_load_obs_mouse
START TEST ( 2023-07-02 14:46:39.542638 ):  test_load_var_mouse
END TEST ( 2023-07-02 14:46:41.049362 ):  test_load_var_mouse
START TEST ( 2023-07-02 14:46:41.051673 ):  test_incremental_read_obs_human
END TEST ( 2023-07-02 14:46:46.651326 ):  test_incremental_read_obs_human
START TEST ( 2023-07-02 14:46:46.653535 ):  test_incremental_read_var_human
END TEST ( 2023-07-02 14:46:48.216871 ):  test_incremental_read_var_human
START TEST ( 2023-07-02 14:46:48.219266 ):  test_incremental_read_obs_mouse
END TEST ( 2023-07-02 14:46:50.634455 ):  test_incremental_read_obs_mouse
START TEST ( 2023-07-02 14:46:50.636518 ):  test_incremental_read_var_mouse
END TEST ( 2023-07-02 14:46:52.02957 ):  test_incremental_read_var_mouse
START TEST ( 2023-07-02 14:46:52.031717 ):  test_incremental_read_X_human
END TEST ( 2023-07-02 15:06:21.675927 ):  test_incremental_read_X_human
START TEST ( 2023-07-02 15:06:21.678379 ):  test_incremental_read_X_human-large-buffer-size
END TEST ( 2023-07-02 15:38:51.28431 ):  test_incremental_read_X_human-large-buffer-size
START TEST ( 2023-07-02 15:38:51.361892 ):  test_incremental_read_X_mouse
END TEST ( 2023-07-02 15:43:56.700087 ):  test_incremental_read_X_mouse
START TEST ( 2023-07-02 15:43:56.720547 ):  test_incremental_read_X_mouse-large-buffer-size
END TEST ( 2023-07-02 15:45:23.18604 ):  test_incremental_read_X_mouse-large-buffer-size
START TEST ( 2023-07-02 15:45:23.188516 ):  test_incremental_query_human_brain
END TEST ( 2023-07-02 15:46:27.33182 ):  test_incremental_query_human_brain
START TEST ( 2023-07-02 15:46:27.333765 ):  test_incremental_query_human_aorta
END TEST ( 2023-07-02 15:46:40.686538 ):  test_incremental_query_human_aorta
START TEST ( 2023-07-02 15:46:40.688573 ):  test_incremental_query_mouse_brain
END TEST ( 2023-07-02 15:46:51.875727 ):  test_incremental_query_mouse_brain
START TEST ( 2023-07-02 15:46:51.877772 ):  test_incremental_query_mouse_aorta
END TEST ( 2023-07-02 15:46:58.295933 ):  test_incremental_query_mouse_aorta
START TEST ( 2023-07-02 15:46:58.29842 ):  test_seurat_small-query
END TEST ( 2023-07-02 15:47:20.06609 ):  test_seurat_small-query
START TEST ( 2023-07-02 15:47:20.067965 ):  test_seurat_10K-cells-human
END TEST ( 2023-07-02 15:47:32.549183 ):  test_seurat_10K-cells-human
START TEST ( 2023-07-02 15:47:32.550956 ):  test_seurat_100K-cells-human
END TEST ( 2023-07-02 15:48:22.766206 ):  test_seurat_100K-cells-human
START TEST ( 2023-07-02 15:48:22.768067 ):  test_seurat_250K-cells-human
END TEST ( 2023-07-02 15:50:07.128338 ):  test_seurat_250K-cells-human
START TEST ( 2023-07-02 15:50:07.130188 ):  test_seurat_500K-cells-human
END TEST ( 2023-07-02 15:53:52.198963 ):  test_seurat_500K-cells-human
START TEST ( 2023-07-02 15:53:52.200954 ):  test_seurat_750K-cells-human
END TEST ( 2023-07-02 15:59:06.944844 ):  test_seurat_750K-cells-human
START TEST ( 2023-07-02 15:59:06.946713 ):  test_seurat_1M-cells-human
END TEST ( 2023-07-02 15:59:50.717664 ):  test_seurat_1M-cells-human
START TEST ( 2023-07-02 15:59:50.720414 ):  test_seurat_common-tissue
END TEST ( 2023-07-02 16:03:47.743072 ):  test_seurat_common-tissue
START TEST ( 2023-07-02 16:03:47.745073 ):  test_seurat_common-tissue-large-buffer-size
END TEST ( 2023-07-02 16:07:50.285648 ):  test_seurat_common-tissue-large-buffer-size
START TEST ( 2023-07-02 16:07:50.287765 ):  test_seurat_common-cell-type
END TEST ( 2023-07-02 16:22:28.235837 ):  test_seurat_common-cell-type
START TEST ( 2023-07-02 16:22:28.241764 ):  test_seurat_common-cell-type-large-buffer-size
END TEST ( 2023-07-02 17:38:56.592504 ):  test_seurat_common-cell-type-large-buffer-size
START TEST ( 2023-07-02 17:38:56.5975 ):  test_seurat_whole-enchilada-large-buffer-size
END TEST ( 2023-07-02 17:38:56.60277 ):  test_seurat_whole-enchilada-large-buffer-size
```

- `acceptance-tests-logs-2023-07-02.csv`

```text
test,user,system,real,test_result
test_load_obs_human,16.559,95.552,7.053,expect_true(nrow(obs_df) > 0): expectation_success: nrow(obs_df) > 0 is not TRUE
test_load_var_human,0.413,0.480000000000004,1.697,expect_true(nrow(var_df) > 0): expectation_success: nrow(var_df) > 0 is not TRUE
test_load_obs_mouse,2.223,7.28399999999999,1.987,expect_true(nrow(obs_df) > 0): expectation_success: nrow(obs_df) > 0 is not TRUE
test_load_var_mouse,0.413,0.475000000000009,1.505,expect_true(nrow(var_df) > 0): expectation_success: nrow(var_df) > 0 is not TRUE
test_incremental_read_obs_human,18.338,103.328,5.598,expect_true(table_iter_is_ok(obs_iter)): expectation_success: table_iter_is_ok(obs_iter) is not TRUE
test_incremental_read_var_human,0.408000000000001,0.482000000000028,1.562,expect_true(table_iter_is_ok(var_iter)): expectation_success: table_iter_is_ok(var_iter) is not TRUE
test_incremental_read_obs_mouse,2.351,6.16499999999999,2.415,expect_true(table_iter_is_ok(obs_iter)): expectation_success: table_iter_is_ok(obs_iter) is not TRUE
test_incremental_read_var_mouse,0.395000000000003,0.401999999999987,1.392,expect_true(table_iter_is_ok(var_iter)): expectation_success: table_iter_is_ok(var_iter) is not TRUE
test_incremental_read_X_human,8479.745,13912.628,1169.643,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_human-large-buffer-size,8247.498,77383.086,1949.527,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_mouse,960.691000000003,1551.99800000001,305.317,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_mouse-large-buffer-size,961.543999999998,1518.712,86.4630000000002,expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_query_human_brain,228.550999999999,269.589999999997,64.1419999999998,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_incremental_query_human_aorta,19.0260000000017,72.7629999999917,13.3519999999999,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_incremental_query_mouse_brain,46.9169999999976,56.8659999999945,11.1860000000001,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_incremental_query_mouse_aorta,12.260000000002,11.2419999999984,6.41700000000037,expect_true(table_iter_is_ok(query$obs())): expectation_success: table_iter_is_ok(query$obs()) is not TRUE ; expect_true(table_iter_is_ok(query$var())): expectation_success: table_iter_is_ok(query$var()) is not TRUE ; expect_true(table_iter_is_ok(query$X("raw")$tables())): expectation_success: table_iter_is_ok(query$X("raw")$tables()) is not TRUE
test_seurat_small-query,25.7360000000008,70.429999999993,21.7669999999998,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_10K-cells-human,9.01800000000003,9.08999999999651,12.4810000000002,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_100K-cells-human,54.7450000000026,48.3930000000109,50.2150000000001,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_250K-cells-human,118.756999999998,94.1370000000024,104.36,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_500K-cells-human,253.110000000001,194.872000000003,225.068,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_750K-cells-human,344.812999999998,264.630999999994,314.742999999999,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_1M-cells-human,160.532999999999,234.426000000007,43.7690000000002,test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_common-tissue,387.287,257.248000000007,237.022,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_common-tissue-large-buffer-size,382.359,260.178,242.54,test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_common-cell-type,3359.342,11201.16,877.945000000001,test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_common-cell-type-large-buffer-size,3579.832,33378.649,4588.348,test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_whole-enchilada-large-buffer-size,0.00400000000081491,0,0.00400000000081491,expect_true(TRUE): expectation_success: TRUE is not TRUE
```

### 2023-06-23

- Host: EC2 instance type: `r6id.x32xlarge`, all nvme mounted as swap.
- Uname: Linux ip-172-31-62-52 5.19.0-1026-aws #27~22.04.1-Ubuntu SMP Mon May 22 15:57:16 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Census version

```r
> cellxgene.census::get_census_version_description('latest')
$release_date
[1] ""

$release_build
[1] "2023-06-20"

$soma.uri
[1] "s3://cellxgene-data-public/cell-census/2023-06-20/soma/"

$soma.s3_region
[1] "us-west-2"

$h5ads.uri
[1] "s3://cellxgene-data-public/cell-census/2023-06-20/h5ads/"

$h5ads.s3_region
[1] "us-west-2"

$alias
[1] "latest"

$census_version
[1] "latest"
```

- R session info

```r
> library("cellxgene.census"); sessionInfo()
R version 4.3.0 (2023-04-21)
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Ubuntu 22.04.2 LTS

Matrix products: default
BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0
LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.10.0

locale:
 [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8
 [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8
 [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C
[10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C

time zone: America/Los_Angeles
tzcode source: system (glibc)

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base

other attached packages:
[1] cellxgene.census_0.0.0.9000

loaded via a namespace (and not attached):
 [1] vctrs_0.6.3           httr_1.4.6            cli_3.6.1
 [4] tiledbsoma_0.0.0.9028 rlang_1.1.1           purrr_1.0.1
 [7] assertthat_0.2.1      data.table_1.14.8     jsonlite_1.8.5
[10] glue_1.6.2            bit_4.0.5             triebeard_0.4.1
[13] grid_4.3.0            RcppSpdlog_0.0.13     base64enc_0.1-3
[16] lifecycle_1.0.3       compiler_4.3.0        fs_1.6.2
[19] Rcpp_1.0.10           aws.s3_0.3.21         lattice_0.21-8
[22] digest_0.6.31         R6_2.5.1              tidyselect_1.2.0
[25] curl_5.0.1            magrittr_2.0.3        urltools_1.7.3
[28] Matrix_1.5-4.1        tools_4.3.0           bit64_4.0.5
[31] aws.signature_0.6.0   spdl_0.0.5            arrow_12.0.1
[34] xml2_1.3.4

```

- `stdout.txt`

```text
START TEST ( 2023-06-23 13:55:52.324647 ):  test_load_obs_human
END TEST ( 2023-06-23 13:55:59.362807 ):  test_load_obs_human
START TEST ( 2023-06-23 13:55:59.364576 ):  test_load_var_human
END TEST ( 2023-06-23 13:56:00.939187 ):  test_load_var_human
START TEST ( 2023-06-23 13:56:00.941131 ):  test_load_obs_mouse
END TEST ( 2023-06-23 13:56:03.004462 ):  test_load_obs_mouse
START TEST ( 2023-06-23 13:56:03.006694 ):  test_load_var_mouse
END TEST ( 2023-06-23 13:56:04.727943 ):  test_load_var_mouse
START TEST ( 2023-06-23 13:56:04.730105 ):  test_incremental_read_obs_human
END TEST ( 2023-06-23 13:56:10.508174 ):  test_incremental_read_obs_human
START TEST ( 2023-06-23 13:56:10.51139 ):  test_incremental_read_var_human
END TEST ( 2023-06-23 13:56:12.073684 ):  test_incremental_read_var_human
START TEST ( 2023-06-23 13:56:12.07612 ):  test_incremental_read_X_human
END TEST ( 2023-06-23 14:17:46.233498 ):  test_incremental_read_X_human
START TEST ( 2023-06-23 14:17:46.236098 ):  test_incremental_read_X_human-large-buffer-size
END TEST ( 2023-06-23 14:42:27.219841 ):  test_incremental_read_X_human-large-buffer-size
START TEST ( 2023-06-23 14:42:27.313921 ):  test_incremental_read_obs_mouse
END TEST ( 2023-06-23 14:42:35.792303 ):  test_incremental_read_obs_mouse
START TEST ( 2023-06-23 14:42:35.825136 ):  test_incremental_read_var_mouse
END TEST ( 2023-06-23 14:44:48.181343 ):  test_incremental_read_var_mouse
START TEST ( 2023-06-23 14:44:48.225299 ):  test_incremental_read_X_mouse
END TEST ( 2023-06-23 14:46:40.709836 ):  test_incremental_read_X_mouse
START TEST ( 2023-06-23 14:46:40.712315 ):  test_incremental_read_X_mouse-large-buffer-size
END TEST ( 2023-06-23 14:48:02.087424 ):  test_incremental_read_X_mouse-large-buffer-size
START TEST ( 2023-06-23 14:48:02.091451 ):  test_incremental_query
END TEST ( 2023-06-23 14:48:02.100564 ):  test_incremental_query
START TEST ( 2023-06-23 14:48:02.102893 ):  test_seurat_small-query
END TEST ( 2023-06-23 14:48:48.744465 ):  test_seurat_small-query
START TEST ( 2023-06-23 14:48:48.746438 ):  test_seurat_10K-cells-human
END TEST ( 2023-06-23 14:49:01.11209 ):  test_seurat_10K-cells-human
START TEST ( 2023-06-23 14:49:01.114074 ):  test_seurat_100K-cells-human
END TEST ( 2023-06-23 14:49:51.088361 ):  test_seurat_100K-cells-human
START TEST ( 2023-06-23 14:49:51.090358 ):  test_seurat_250K-cells-human
END TEST ( 2023-06-23 14:51:32.084494 ):  test_seurat_250K-cells-human
START TEST ( 2023-06-23 14:51:32.086453 ):  test_seurat_500K-cells-human
END TEST ( 2023-06-23 14:55:04.211365 ):  test_seurat_500K-cells-human
START TEST ( 2023-06-23 14:55:04.213284 ):  test_seurat_750K-cells-human
END TEST ( 2023-06-23 15:00:02.888813 ):  test_seurat_750K-cells-human
START TEST ( 2023-06-23 15:00:02.890819 ):  test_seurat_1M-cells-human
END TEST ( 2023-06-23 15:00:54.993723 ):  test_seurat_1M-cells-human
START TEST ( 2023-06-23 15:00:54.996504 ):  test_seurat_common-tissue
END TEST ( 2023-06-23 15:05:05.376396 ):  test_seurat_common-tissue
START TEST ( 2023-06-23 15:05:05.378534 ):  test_seurat_common-tissue-large-buffer-size
END TEST ( 2023-06-23 15:09:13.780464 ):  test_seurat_common-tissue-large-buffer-size
START TEST ( 2023-06-23 15:09:13.782572 ):  test_seurat_common-cell-type
END TEST ( 2023-06-23 15:24:43.865822 ):  test_seurat_common-cell-type
START TEST ( 2023-06-23 15:24:43.867832 ):  test_seurat_common-cell-type-large-buffer-size
END TEST ( 2023-06-23 16:56:58.016858 ):  test_seurat_common-cell-type-large-buffer-size
START TEST ( 2023-06-23 16:56:58.020263 ):  test_seurat_whole-enchilada-large-buffer-size
END TEST ( 2023-06-23 16:56:58.025497 ):  test_seurat_whole-enchilada-large-buffer-size
```

- `acceptance-tests-logs-2023-06-23.csv`

```text
test,user,system,real,test_result
test_load_obs_human,18.366,101.34,7.037,expect_true(nrow(obs_df) > 0): expectation_success: nrow(obs_df) > 0 is not TRUE
test_load_var_human,0.369,0.530999999999992,1.569,expect_true(nrow(var_df) > 0): expectation_success: nrow(var_df) > 0 is not TRUE
test_load_obs_mouse,2.011,6.202,2.062,expect_true(nrow(obs_df) > 0): expectation_success: nrow(obs_df) > 0 is not TRUE
test_load_var_mouse,0.417999999999999,0.486000000000004,1.721,expect_true(nrow(var_df) > 0): expectation_success: nrow(var_df) > 0 is not TRUE
test_incremental_read_obs_human,18.411,94.142,5.777,expect_true(table_iter_is_ok(obs_iter)): expectation_success: table_iter_is_ok(obs_iter) is not TRUE
test_incremental_read_var_human,0.454999999999998,0.531000000000006,1.561,expect_true(table_iter_is_ok(var_iter)): expectation_success: table_iter_is_ok(var_iter) is not TRUE
test_incremental_read_X_human,8193.649,14613.005,1294.156,expect_warning(X_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$X$get("raw")$read()$tables()): expectation_success:  ; expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_human-large-buffer-size,8091.321,43302.612,1480.899,expect_warning(X_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$X$get("raw")$read()$tables()): expectation_success:  ; expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_obs_mouse,2.22500000000036,13.4630000000034,8.45900000000029,expect_true(table_iter_is_ok(obs_iter)): expectation_success: table_iter_is_ok(obs_iter) is not TRUE
test_incremental_read_var_mouse,0.7549999999992,128.547999999995,132.348,expect_true(table_iter_is_ok(var_iter)): expectation_success: table_iter_is_ok(var_iter) is not TRUE
test_incremental_read_X_mouse,928.090000000002,1636.642,112.482,expect_warning(X_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$X$get("raw")$read()$tables()): expectation_success:  ; expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_read_X_mouse-large-buffer-size,899.965,1779.616,81.373,expect_warning(X_iter <- census$get("census_data")$get(organism)$ms$get("RNA")$X$get("raw")$read()$tables()): expectation_success:  ; expect_true(table_iter_is_ok(X_iter)): expectation_success: table_iter_is_ok(X_iter) is not TRUE
test_incremental_query,0.00500000000101863,0.000999999996565748,0.00800000000026557,expect_true(TRUE): expectation_success: TRUE is not TRUE
test_seurat_small-query,26.2459999999992,105.765999999996,46.6410000000001,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_10K-cells-human,8.78700000000026,9.39099999999598,12.3649999999998,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_100K-cells-human,52.6959999999999,51.7980000000025,49.973,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_250K-cells-human,117.172999999999,100.030000000006,100.993,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_500K-cells-human,236.339,191.728999999999,212.124,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_750K-cells-human,334.721999999998,264.296999999999,298.675,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_1M-cells-human,156.661,246.656000000003,52.1020000000003,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_common-tissue,392.353999999999,277.388999999996,250.379,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_common-tissue-large-buffer-size,379.781999999999,300.778000000006,248.401000000001,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): expectation_success: is(this_seurat, "Seurat") is not TRUE ; test_seurat(test_args): expectation_success: ncol(this_seurat) > 0 is not TRUE
test_seurat_common-cell-type,3346.826,11816.851,930.083,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_common-cell-type-large-buffer-size,3437.291,28305.89,5534.148,test_seurat(test_args): expectation_warning: Iteration results cannot be concatenated on its entirety because array has non-zero elements greater than '.Machine$integer.max'. ; test_seurat(test_args): Error: Error in `vec_to_Array(x, type)`: long vectors not supported yet: memory.c:3888
test_seurat_whole-enchilada-large-buffer-size,0.00399999999717693,0.00099999998928979,0.00500000000101863,expect_true(TRUE): expectation_success: TRUE is not TRUE
```
