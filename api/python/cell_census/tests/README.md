# Test README

This directory contains tests of the cell-census package API, _and_ the use of the API on the
live "corpus", i.e., data in the public cell census S3 bucket. The tests use Pytest, and have
Pytest marks to control which tests are run.

Tests can be run in the usual manner. First, ensure you have cell-census installed, e.g., from the top-level repo directory:

> pip install -e ./api/python/cell_census/

Then run the tests:

> pytest ./api/python/cell_census/

## Pytest Marks

There are two Pytest marks you can use from the command line:

- live_corpus: tests that directly access the `latest` version of the Cell Census. Enabled by default.
- expensive: tests that are expensive (ie., cpu, memory, time). Disabled by default - enable with `--expensive`. Some of these tests are _very_ expensive, ie., require a very large memory host to succeed.

By default, only relatively cheap & fast tests are run. To enable `expensive` tests:

> pytest --expensive ...

To disable `live_corpus` tests:

> pytest -m 'not live_corpus'

You can also combine them, e.g.,

> pytest -m 'not live_corpus' --expensive

# Acceptance (expensive) tests

These tests are periodically run, and are not part of CI due to their overhead.

When run, please record the results in this file (below) and commit the change to git. Please include the following information:

- date
- config:
  - EC2 instance type and any system config (i.e., swap)
  - host and OS as reported by `uname -a`
  - Python & package versions and OS - suggest capturing the output of `tiledbsoma.show_package_versions()`
  - The Cell Census version used for the test (i.e., the version aliased as `latest`). This can be easily captured using `cell_census.get_census_version_description('latest')`
- any run notes
- full output of: `pytest -v --durations=0 --expensive ./api/python/cell_census/tests/`

## 2023-03-28

**Config**

- Host: EC2 instance type: `r6id.x32xlarge`, all nvme mounted as swap.
- Uname: Linux bruce.aegea 5.15.0-1033-aws #37~20.04.1-Ubuntu SMP Fri Mar 17 11:39:30 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Python & census versions:

```
In [3]: tiledbsoma.show_package_versions()
tiledbsoma.__version__        1.2.1
TileDB-Py tiledb.version()    (0, 21, 1)
TileDB core version           2.15.0
libtiledbsoma version()       libtiledbsoma=;libtiledb=2.15.0
python version                3.9.16.final.0
OS version                    Linux 5.15.0-1033-aws

In [4]: cell_census.get_census_version_description('latest')
Out[4]:
{'release_date': None,
'release_build': '2023-03-16',
'soma': {'uri': 's3://cellxgene-data-public/cell-census/2023-03-16/soma/',
  's3_region': 'us-west-2'},
'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2023-03-16/h5ads/',
  's3_region': 'us-west-2'}}
```

**Run notes:**

The test `test_acceptance.py::test_get_anndata[None-homo_sapiens]` manifest a large amount of paging activity.

**Pytest output:**

```
$ pytest -v --durations=0 --expensive ./api/python/cell_census/tests/
========================================================== test session starts ===========================================================
platform linux -- Python 3.9.16, pytest-7.2.2, pluggy-1.0.0 -- /home/bruce/cell-census/venv/bin/python
cachedir: .pytest_cache
rootdir: /home/bruce/cell-census/api/python/cell_census, configfile: pyproject.toml
plugins: requests-mock-1.10.0, anyio-3.6.2
collected 39 items

api/python/cell_census/tests/test_acceptance.py::test_load_axes[homo_sapiens] PASSED                                               [  2%]
api/python/cell_census/tests/test_acceptance.py::test_load_axes[mus_musculus] PASSED                                               [  5%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_read[homo_sapiens] PASSED                                        [  7%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_read[mus_musculus] PASSED                                        [ 10%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens] PASSED                     [ 12%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus] PASSED                     [ 15%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens] PASSED                     [ 17%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus] PASSED                     [ 20%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens] PASSED                  [ 23%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus] PASSED                  [ 25%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens] PASSED                  [ 28%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus] PASSED                  [ 30%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue == 'aorta'-ctx_config0-homo_sapiens] PASSED               [ 33%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue == 'aorta'-ctx_config0-mus_musculus] PASSED               [ 35%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[cell_type == 'neuron'-ctx_config1-homo_sapiens] PASSED           [ 38%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[cell_type == 'neuron'-ctx_config1-mus_musculus] PASSED           [ 41%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue == 'brain'-ctx_config2-homo_sapiens] PASSED               [ 43%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue == 'brain'-ctx_config2-mus_musculus] PASSED               [ 46%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[is_primary_data == True-ctx_config3-homo_sapiens] PASSED         [ 48%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[is_primary_data == True-ctx_config3-mus_musculus] PASSED         [ 51%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[None-ctx_config4-homo_sapiens] PASSED                            [ 53%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[None-ctx_config4-mus_musculus] PASSED                            [ 56%]
api/python/cell_census/tests/test_directory.py::test_get_census_version_directory PASSED                                           [ 58%]
api/python/cell_census/tests/test_directory.py::test_get_census_version_description_errors PASSED                                  [ 61%]
api/python/cell_census/tests/test_directory.py::test_live_directory_contents PASSED                                                [ 64%]
api/python/cell_census/tests/test_get_anndata.py::test_get_anndata_value_filter PASSED                                             [ 66%]
api/python/cell_census/tests/test_get_anndata.py::test_get_anndata_coords PASSED                                                   [ 69%]
api/python/cell_census/tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter PASSED                         [ 71%]
api/python/cell_census/tests/test_get_helpers.py::test_get_experiment PASSED                                                       [ 74%]
api/python/cell_census/tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens] PASSED                                    [ 76%]
api/python/cell_census/tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus] PASSED                                    [ 79%]
api/python/cell_census/tests/test_open.py::test_open_soma_latest PASSED                                                            [ 82%]
api/python/cell_census/tests/test_open.py::test_open_soma_with_context PASSED                                                      [ 84%]
api/python/cell_census/tests/test_open.py::test_open_soma_errors PASSED                                                            [ 87%]
api/python/cell_census/tests/test_open.py::test_get_source_h5ad_uri PASSED                                                         [ 89%]
api/python/cell_census/tests/test_open.py::test_get_source_h5ad_uri_errors PASSED                                                  [ 92%]
api/python/cell_census/tests/test_open.py::test_download_source_h5ad PASSED                                                        [ 94%]
api/python/cell_census/tests/test_open.py::test_download_source_h5ad_errors PASSED                                                 [ 97%]
api/python/cell_census/tests/test_util.py::test_uri_join PASSED                                                                    [100%]

=========================================================== slowest durations ============================================================
5384.37s call     tests/test_acceptance.py::test_get_anndata[None-ctx_config4-homo_sapiens]
1191.86s call     tests/test_acceptance.py::test_get_anndata[is_primary_data == True-ctx_config3-homo_sapiens]
396.88s call     tests/test_acceptance.py::test_get_anndata[cell_type == 'neuron'-ctx_config1-homo_sapiens]
181.64s call     tests/test_acceptance.py::test_get_anndata[None-ctx_config4-mus_musculus]
107.21s call     tests/test_acceptance.py::test_get_anndata[is_primary_data == True-ctx_config3-mus_musculus]
36.18s call     tests/test_acceptance.py::test_get_anndata[tissue == 'brain'-ctx_config2-homo_sapiens]
27.69s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens]
22.58s call     tests/test_acceptance.py::test_get_anndata[cell_type == 'neuron'-ctx_config1-mus_musculus]
20.50s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens]
12.39s call     tests/test_acceptance.py::test_get_anndata[tissue == 'brain'-ctx_config2-mus_musculus]
11.72s call     tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
9.52s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus]
8.18s call     tests/test_acceptance.py::test_incremental_read[homo_sapiens]
8.07s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens]
7.25s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens]
7.20s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus]
6.71s call     tests/test_acceptance.py::test_load_axes[homo_sapiens]
6.65s call     tests/test_acceptance.py::test_get_anndata[tissue == 'aorta'-ctx_config0-homo_sapiens]
5.94s setup    tests/test_open.py::test_download_source_h5ad
5.76s call     tests/test_directory.py::test_live_directory_contents
5.09s call     tests/test_open.py::test_get_source_h5ad_uri
4.67s call     tests/test_get_anndata.py::test_get_anndata_value_filter
4.50s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus]
4.13s call     tests/test_open.py::test_open_soma_latest
4.00s call     tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens]
3.91s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus]
3.53s call     tests/test_acceptance.py::test_incremental_read[mus_musculus]
3.48s call     tests/test_acceptance.py::test_get_anndata[tissue == 'aorta'-ctx_config0-mus_musculus]
3.00s call     tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus]
2.83s call     tests/test_get_anndata.py::test_get_anndata_coords
2.42s call     tests/test_open.py::test_download_source_h5ad
2.11s call     tests/test_acceptance.py::test_load_axes[mus_musculus]
1.93s setup    tests/test_get_anndata.py::test_get_anndata_coords
1.20s call     tests/test_open.py::test_get_source_h5ad_uri_errors
0.90s setup    tests/test_open.py::test_download_source_h5ad_errors
0.60s call     tests/test_get_helpers.py::test_get_experiment
0.44s call     tests/test_open.py::test_open_soma_with_context
0.31s setup    tests/test_get_anndata.py::test_get_anndata_value_filter
0.24s setup    tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
0.05s call     tests/test_directory.py::test_get_census_version_description_errors
0.03s setup    tests/test_directory.py::test_get_census_version_directory
0.03s call     tests/test_directory.py::test_get_census_version_directory
0.02s teardown tests/test_acceptance.py::test_get_anndata[None-ctx_config4-homo_sapiens]
0.01s setup    tests/test_acceptance.py::test_get_anndata[None-ctx_config4-mus_musculus]
0.01s setup    tests/test_acceptance.py::test_get_anndata[is_primary_data == True-ctx_config3-mus_musculus]
0.01s teardown tests/test_acceptance.py::test_get_anndata[is_primary_data == True-ctx_config3-homo_sapiens]

(71 durations < 0.005s hidden.  Use -vv to show these durations.)
==================================================== 39 passed in 7508.82s (2:05:08) =====================================================
```
