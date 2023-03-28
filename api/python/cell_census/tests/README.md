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

The test `test_acceptance.py::test_get_anndata[None-homo_sapiens]` manifest a large amount of paging activity. It seems likely that memory use has increased significantly since the past (annecdotal) run - I believe this used to run in ~600 seconds on this same EC2 instance type/config.

**Pytest output:**

```
$ pytest -v --durations=0 --expensive ./api/python/cell_census/tests/
=============================================================== test session starts ===============================================================
platform linux -- Python 3.9.16, pytest-7.2.2, pluggy-1.0.0 -- /home/bruce/cell-census/venv/bin/python
cachedir: .pytest_cache
rootdir: /home/bruce/cell-census/api/python/cell_census, configfile: pyproject.toml
plugins: requests-mock-1.10.0, anyio-3.6.2
collected 37 items

api/python/cell_census/tests/test_acceptance.py::test_load_axes[homo_sapiens] PASSED                                                        [  2%]
api/python/cell_census/tests/test_acceptance.py::test_load_axes[mus_musculus] PASSED                                                        [  5%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_read[homo_sapiens] PASSED                                                 [  8%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_read[mus_musculus] PASSED                                                 [ 10%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens] PASSED                              [ 13%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus] PASSED                              [ 16%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens] PASSED                              [ 18%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus] PASSED                              [ 21%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens] PASSED                           [ 24%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus] PASSED                           [ 27%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens] PASSED                           [ 29%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus] PASSED                           [ 32%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue == 'aorta'-homo_sapiens] PASSED                                    [ 35%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue == 'aorta'-mus_musculus] PASSED                                    [ 37%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[cell_type == 'neuron'-homo_sapiens] PASSED                                [ 40%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[cell_type == 'neuron'-mus_musculus] PASSED                                [ 43%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue == 'brain'-homo_sapiens] PASSED                                    [ 45%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue == 'brain'-mus_musculus] PASSED                                    [ 48%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[None-homo_sapiens] PASSED                                                 [ 51%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[None-mus_musculus] PASSED                                                 [ 54%]
api/python/cell_census/tests/test_directory.py::test_get_census_version_directory PASSED                                                    [ 56%]
api/python/cell_census/tests/test_directory.py::test_get_census_version_description_errors PASSED                                           [ 59%]
api/python/cell_census/tests/test_directory.py::test_live_directory_contents PASSED                                                         [ 62%]
api/python/cell_census/tests/test_get_anndata.py::test_get_anndata_value_filter PASSED                                                      [ 64%]
api/python/cell_census/tests/test_get_anndata.py::test_get_anndata_coords PASSED                                                            [ 67%]
api/python/cell_census/tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter PASSED                                  [ 70%]
api/python/cell_census/tests/test_get_helpers.py::test_get_experiment PASSED                                                                [ 72%]
api/python/cell_census/tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens] PASSED                                             [ 75%]
api/python/cell_census/tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus] PASSED                                             [ 78%]
api/python/cell_census/tests/test_open.py::test_open_soma_latest PASSED                                                                     [ 81%]
api/python/cell_census/tests/test_open.py::test_open_soma_with_context PASSED                                                               [ 83%]
api/python/cell_census/tests/test_open.py::test_open_soma_errors PASSED                                                                     [ 86%]
api/python/cell_census/tests/test_open.py::test_get_source_h5ad_uri PASSED                                                                  [ 89%]
api/python/cell_census/tests/test_open.py::test_get_source_h5ad_uri_errors PASSED                                                           [ 91%]
api/python/cell_census/tests/test_open.py::test_download_source_h5ad PASSED                                                                 [ 94%]
api/python/cell_census/tests/test_open.py::test_download_source_h5ad_errors PASSED                                                          [ 97%]
api/python/cell_census/tests/test_util.py::test_uri_join PASSED                                                                             [100%]

=========================================================++++++=== slowest durations ==============================================================
5552.78s call     tests/test_acceptance.py::test_get_anndata[None-homo_sapiens]
391.00s call     tests/test_acceptance.py::test_get_anndata[cell_type == 'neuron'-homo_sapiens]
215.23s call     tests/test_acceptance.py::test_get_anndata[None-mus_musculus]
35.67s call     tests/test_acceptance.py::test_get_anndata[tissue == 'brain'-homo_sapiens]
26.97s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens]
24.33s call     tests/test_acceptance.py::test_get_anndata[cell_type == 'neuron'-mus_musculus]
20.07s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens]
11.90s call     tests/test_acceptance.py::test_get_anndata[tissue == 'brain'-mus_musculus]
10.74s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus]
9.41s call     tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
9.39s call     tests/test_acceptance.py::test_get_anndata[tissue == 'aorta'-homo_sapiens]
9.09s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens]
8.77s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus]
7.63s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens]
7.17s call     tests/test_acceptance.py::test_incremental_read[homo_sapiens]
7.02s call     tests/test_acceptance.py::test_load_axes[homo_sapiens]
5.40s setup    tests/test_open.py::test_download_source_h5ad
5.31s call     tests/test_directory.py::test_live_directory_contents
5.00s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus]
4.91s call     tests/test_get_anndata.py::test_get_anndata_value_filter
4.90s call     tests/test_open.py::test_get_source_h5ad_uri
4.58s call     tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus]
4.56s call     tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens]
4.35s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus]
4.04s call     tests/test_acceptance.py::test_get_anndata[tissue == 'aorta'-mus_musculus]
3.04s call     tests/test_acceptance.py::test_incremental_read[mus_musculus]
2.54s call     tests/test_get_anndata.py::test_get_anndata_coords
2.33s call     tests/test_open.py::test_download_source_h5ad
2.24s call     tests/test_acceptance.py::test_load_axes[mus_musculus]
1.34s call     tests/test_open.py::test_get_source_h5ad_uri_errors
1.27s setup    tests/test_get_anndata.py::test_get_anndata_coords
0.97s setup    tests/test_open.py::test_download_source_h5ad_errors
0.48s call     tests/test_open.py::test_open_soma_latest
0.48s call     tests/test_open.py::test_open_soma_with_context
0.43s call     tests/test_get_helpers.py::test_get_experiment
0.31s setup    tests/test_get_anndata.py::test_get_anndata_value_filter
0.23s setup    tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
0.09s call     tests/test_directory.py::test_get_census_version_description_errors
0.04s setup    tests/test_directory.py::test_get_census_version_directory
0.02s teardown tests/test_acceptance.py::test_get_anndata[None-homo_sapiens]
0.02s call     tests/test_directory.py::test_get_census_version_directory
0.01s setup    tests/test_acceptance.py::test_get_anndata[None-mus_musculus]

(69 durations < 0.005s hidden.  Use -vv to show these durations.)
========================================================== 37 passed in 6407.08s (1:46:47) ===========================================================
```
