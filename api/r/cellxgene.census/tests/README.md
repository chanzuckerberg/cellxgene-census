# Test README

This directory contains tests of the `cellxgene.census` R package API, _and_ the use of the API on the live "corpus", i.e., data in the public Census S3 bucket. The tests use the R package `tessthat`.

In addition, a set of acceptance (expensive) tests are available and `testthat` does not run them by default (see [section below](#Acceptance-expensive-tests)).

Tests can be run in the usual manner. First, ensure you have `cellxgene-census` and `testthat` installed, e.g., from the top-level repo directory:

Then run the tests from R in the repo root folder:

```r
library("testthat")
library("cellxgene.census")
test_dir("./api/r/cellxgene.census/tests/")
```

# Acceptance (expensive) tests

These tests are periodically run, and are not part of CI due to their overhead.

These tests use a modified `Reporter` from `testthat` to record running times of each test in a `csv` file. To run the test execute the following command:

```
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


## 2023-07-02

- Host: EC2 instance type: `r6id.x32xlarge`, all nvme mounted as swap.
- Uname: Linux ip-172-31-62-52 5.19.0-1028-aws #29~22.04.1-Ubuntu SMP Tue Jun 20 19:12:11 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Census version

```
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

```
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

```
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

```
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


## 2023-06-23

- Host: EC2 instance type: `r6id.x32xlarge`, all nvme mounted as swap.
- Uname: Linux ip-172-31-62-52 5.19.0-1026-aws #27~22.04.1-Ubuntu SMP Mon May 22 15:57:16 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Census version

```
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

```
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

```
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

- `acceptance-tests-logs-2023-06-23.csv `

```
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