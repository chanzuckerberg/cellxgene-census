# Test README

This directory contains tests of the cellxgene-census package API, _and_ the use of the API on the
live "corpus", i.e., data in the public Census S3 bucket. The tests use Pytest, and have
Pytest marks to control which tests are run.

Tests can be run in the usual manner. First, ensure you have cellxgene-census installed, e.g., from the top-level repo directory:

> pip install -e ./api/python/cellxgene_census/

Then run the tests:

> pytest ./api/python/cellxgene_census/

## Pytest Marks

There are various Pytest marks you can use from the command line:

- live_corpus: tests that directly access the `latest` version of the Census. Enabled by default.
- expensive: tests that are expensive (ie., cpu, memory, time). Disabled by default - enable with `--expensive`. Some of these tests are _very_ expensive, ie., require a very large memory host to succeed.
- experimental: tests that are for code in the `experimental` package. Disabled by default - enable with `--experimental`. These tests require installation the optional Python packages installed via pip `pip install -e ./api/python/cellxgene_census/[experimental]`

By default, only relatively cheap & fast tests are run. To enable `expensive` tests:

> pytest --expensive ...

To enable `experimental` tests:

> pytest --experimental ...

To disable `live_corpus` tests:

> pytest -m 'not live_corpus'

You can also combine them, e.g.,

> pytest -m 'not live_corpus' --expensive --experimental

# Acceptance (expensive) tests

These tests are periodically run, and are not part of CI due to their overhead.

When run, please record the results in this file (below) and commit the change to git. Please include the following information:

- date
- config:
  - EC2 instance type and any system config (i.e., swap)
  - host and OS as reported by `uname -a`
  - Python & package versions and OS - suggest capturing the output of `tiledbsoma.show_package_versions()`
  - The Census version used for the test (i.e., the version aliased as `latest`). This can be easily captured using `cellxgene_census.get_census_version_description('latest')`
  - the cellxgene_census package version (ie., `cellxgene_census.__version__`)
- any run notes
- full output of: `pytest -v --durations=0 --expensive ./api/python/cellxgene_census/tests/`

## 2023-05-16

- Host: EC2 instance type: `r6id.x32xlarge`, all nvme mounted as swap.
- Uname: Linux ip-172-31-59-194 5.19.0-1022-aws #23~22.04.1-Ubuntu SMP Fri Mar 17 15:38:24 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Python & census versions:
```
>>> import cellxgene_census, tiledbsoma
>>> tiledbsoma.show_package_versions()
tiledbsoma.__version__        1.2.3
TileDB-Py tiledb.version()    (0, 21, 3)
TileDB core version           2.15.2
libtiledbsoma version()       libtiledb=2.15.2
python version                3.10.6.final.0
OS version                    Linux 5.19.0-1022-aws
>>> cellxgene_census.__version__
  '1.0.2.dev2+g1598cfd'
>>> cellxgene_census.get_census_version_description('latest')
{'release_date': None, 'release_build': '2023-05-15', 'soma': {'uri': 's3://cellxgene-data-public/cell-census/2023-05-15/soma/', 's3_region': 'us-west-2'}, 'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2023-05-15/h5ads/', 's3_region': 'us-west-2'}, 'alias': 'latest'}
```

**Pytest output:**

```
============================= test session starts ==============================
platform linux -- Python 3.10.6, pytest-7.3.1, pluggy-1.0.0 -- /home/ubuntu/venv-cellxgene-census/bin/python3
cachedir: .pytest_cache
rootdir: /home/ubuntu/cellxgene-census/api/python/cellxgene_census
configfile: pyproject.toml
plugins: requests-mock-1.10.0
collecting ... collected 51 items

tests/test_acceptance.py::test_load_axes[homo_sapiens] PASSED            [  1%]
tests/test_acceptance.py::test_load_axes[mus_musculus] PASSED            [  3%]
tests/test_acceptance.py::test_incremental_read[homo_sapiens] PASSED     [  5%]
tests/test_acceptance.py::test_incremental_read[mus_musculus] PASSED     [  7%]
tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens] PASSED [  9%]
tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus] PASSED [ 11%]
tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens] PASSED [ 13%]
tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus] PASSED [ 15%]
tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens] PASSED [ 17%]
tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus] PASSED [ 19%]
tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens] PASSED [ 21%]
tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus] PASSED [ 23%]
tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-homo_sapiens] PASSED [ 25%]
tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-mus_musculus] PASSED [ 27%]
tests/test_acceptance.py::test_get_anndata[First 10K cells-homo_sapiens] PASSED [ 29%]
tests/test_acceptance.py::test_get_anndata[First 10K cells-mus_musculus] PASSED [ 31%]
tests/test_acceptance.py::test_get_anndata[First 100K cells-homo_sapiens] PASSED [ 33%]
tests/test_acceptance.py::test_get_anndata[First 100K cells-mus_musculus] PASSED [ 35%]
tests/test_acceptance.py::test_get_anndata[First 1M cells-homo_sapiens] PASSED [ 37%]
tests/test_acceptance.py::test_get_anndata[First 1M cells-mus_musculus] PASSED [ 39%]
tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config4-homo_sapiens] PASSED [ 41%]
tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config4-mus_musculus] PASSED [ 43%]
tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config5-homo_sapiens] PASSED [ 45%]
tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config5-mus_musculus] PASSED [ 47%]
tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-homo_sapiens] PASSED [ 49%]
tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-mus_musculus] PASSED [ 50%]
tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-homo_sapiens] PASSED [ 52%]
tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-mus_musculus] PASSED [ 54%]
tests/test_directory.py::test_get_census_version_directory PASSED        [ 56%]
tests/test_directory.py::test_get_census_version_description_errors PASSED [ 58%]
tests/test_directory.py::test_live_directory_contents PASSED             [ 60%]
tests/test_get_anndata.py::test_get_anndata_value_filter PASSED          [ 62%]
tests/test_get_anndata.py::test_get_anndata_coords PASSED                [ 64%]
tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter PASSED [ 66%]
tests/test_get_helpers.py::test_get_experiment PASSED                    [ 68%]
tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens] PASSED [ 70%]
tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus] PASSED [ 72%]
tests/test_open.py::test_open_soma_stable PASSED                         [ 74%]
tests/test_open.py::test_open_soma_latest PASSED                         [ 76%]
tests/test_open.py::test_open_soma_with_context PASSED                   [ 78%]
tests/test_open.py::test_open_soma_invalid_args PASSED                   [ 80%]
tests/test_open.py::test_open_soma_errors PASSED                         [ 82%]
tests/test_open.py::test_open_soma_defaults_to_latest_if_missing_stable PASSED [ 84%]
tests/test_open.py::test_open_soma_defaults_to_stable PASSED             [ 86%]
tests/test_open.py::test_get_source_h5ad_uri PASSED                      [ 88%]
tests/test_open.py::test_get_source_h5ad_uri_errors PASSED               [ 90%]
tests/test_open.py::test_download_source_h5ad PASSED                     [ 92%]
tests/test_open.py::test_download_source_h5ad_errors PASSED              [ 94%]
tests/test_open.py::test_opening_census_without_anon_access_fails_with_bogus_creds PASSED [ 96%]
tests/test_open.py::test_can_open_with_anonymous_access PASSED           [ 98%]
tests/test_util.py::test_uri_join PASSED                                 [100%]

============================== slowest durations ===============================
6905.70s call     tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-homo_sapiens]
2222.81s call     tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-homo_sapiens]
743.26s call     tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config4-homo_sapiens]
223.36s call     tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-mus_musculus]
174.85s call     tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-mus_musculus]
51.53s call     tests/test_acceptance.py::test_get_anndata[First 1M cells-mus_musculus]
50.58s call     tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config5-homo_sapiens]
39.31s call     tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config4-mus_musculus]
37.15s call     tests/test_acceptance.py::test_get_anndata[First 1M cells-homo_sapiens]
34.09s call     tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config5-mus_musculus]
30.23s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens]
19.14s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens]
14.76s call     tests/test_directory.py::test_live_directory_contents
13.29s call     tests/test_acceptance.py::test_get_anndata[First 100K cells-mus_musculus]
9.64s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus]
9.48s call     tests/test_acceptance.py::test_get_anndata[First 100K cells-homo_sapiens]
9.48s call     tests/test_acceptance.py::test_incremental_read[homo_sapiens]
9.47s call     tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
8.13s call     tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-homo_sapiens]
7.94s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus]
7.82s call     tests/test_acceptance.py::test_load_axes[homo_sapiens]
7.52s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens]
7.32s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens]
5.51s call     tests/test_acceptance.py::test_get_anndata[First 10K cells-homo_sapiens]
5.44s call     tests/test_get_anndata.py::test_get_anndata_value_filter
4.89s call     tests/test_acceptance.py::test_get_anndata[First 10K cells-mus_musculus]
4.84s call     tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-mus_musculus]
4.73s call     tests/test_acceptance.py::test_incremental_read[mus_musculus]
4.17s call     tests/test_open.py::test_get_source_h5ad_uri
3.92s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus]
3.59s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus]
3.29s call     tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens]
3.11s call     tests/test_get_anndata.py::test_get_anndata_coords
2.58s call     tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus]
1.99s call     tests/test_acceptance.py::test_load_axes[mus_musculus]
1.89s call     tests/test_open.py::test_download_source_h5ad
1.06s call     tests/test_open.py::test_open_soma_with_context
1.05s setup    tests/test_open.py::test_download_source_h5ad_errors
1.00s call     tests/test_open.py::test_open_soma_stable
0.96s call     tests/test_open.py::test_get_source_h5ad_uri_errors
0.89s setup    tests/test_open.py::test_download_source_h5ad
0.75s call     tests/test_get_helpers.py::test_get_experiment
0.42s setup    tests/test_get_anndata.py::test_get_anndata_value_filter
0.39s call     tests/test_open.py::test_open_soma_defaults_to_latest_if_missing_stable
0.34s setup    tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
0.34s call     tests/test_open.py::test_can_open_with_anonymous_access
0.33s call     tests/test_open.py::test_open_soma_latest
0.32s call     tests/test_directory.py::test_get_census_version_description_errors
0.31s setup    tests/test_get_anndata.py::test_get_anndata_coords
0.25s call     tests/test_open.py::test_opening_census_without_anon_access_fails_with_bogus_creds
0.04s setup    tests/test_directory.py::test_get_census_version_directory
0.03s call     tests/test_directory.py::test_get_census_version_directory
0.01s teardown tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-homo_sapiens]
0.01s setup    tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-mus_musculus]
0.01s teardown tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-homo_sapiens]
0.01s setup    tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-mus_musculus]

(97 durations < 0.005s hidden.  Use -vv to show these durations.)
======================= 51 passed in 10696.20s (2:58:16) =======================
```

## 2023-03-29

**Config**

- Host: EC2 instance type: `r6id.x32xlarge`, all nvme mounted as swap.
- Uname: Linux bruce.aegea 5.15.0-1033-aws #37~20.04.1-Ubuntu SMP Fri Mar 17 11:39:30 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Python & census versions:

```
In [1]: import cell_census, tiledbsoma

In [2]: tiledbsoma.show_package_versions()
tiledbsoma.__version__        1.2.1
TileDB-Py tiledb.version()    (0, 21, 1)
TileDB core version           2.15.0
libtiledbsoma version()       libtiledbsoma=;libtiledb=2.15.0
python version                3.9.16.final.0
OS version                    Linux 5.15.0-1033-aws

In [3]: cell_census.get_census_version_description('latest')
Out[3]: 
{'release_date': None,
 'release_build': '2023-03-16',
 'soma': {'uri': 's3://cellxgene-data-public/cell-census/2023-03-16/soma/',
  's3_region': 'us-west-2'},
 'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2023-03-16/h5ads/',
  's3_region': 'us-west-2'}}

In [4]: cell_census.__version__
Out[4]: '0.12.0'
```

**Run notes:**

The test `test_acceptance.py::test_get_anndata[None-homo_sapiens]` manifest a large amount of paging activity.

**Pytest output:**

```
$ pytest -v --durations=0 --expensive ./api/python/cell_census/tests/
==================================================== test session starts =====================================================
platform linux -- Python 3.9.16, pytest-7.2.2, pluggy-1.0.0 -- /home/bruce/cell-census/venv/bin/python
cachedir: .pytest_cache
rootdir: /home/bruce/cell-census/api/python/cell_census, configfile: pyproject.toml
plugins: requests-mock-1.10.0, anyio-3.6.2
collected 45 items                                                                                                           

api/python/cell_census/tests/test_acceptance.py::test_load_axes[homo_sapiens] PASSED                                   [  2%]
api/python/cell_census/tests/test_acceptance.py::test_load_axes[mus_musculus] PASSED                                   [  4%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_read[homo_sapiens] PASSED                            [  6%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_read[mus_musculus] PASSED                            [  8%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens] PASSED         [ 11%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus] PASSED         [ 13%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens] PASSED         [ 15%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus] PASSED         [ 17%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens] PASSED      [ 20%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus] PASSED      [ 22%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens] PASSED      [ 24%]
api/python/cell_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus] PASSED      [ 26%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-homo_sapiens] PASSED [ 28%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-mus_musculus] PASSED [ 31%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[First 10K cells-homo_sapiens] PASSED                 [ 33%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[First 10K cells-mus_musculus] PASSED                 [ 35%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[First 100K cells-homo_sapiens] PASSED                [ 37%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[First 100K cells-mus_musculus] PASSED                [ 40%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[First 1M cells-homo_sapiens] PASSED                  [ 42%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[First 1M cells-mus_musculus] PASSED                  [ 44%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config4-homo_sapiens] PASSED [ 46%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config4-mus_musculus] PASSED [ 48%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config5-homo_sapiens] PASSED [ 51%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config5-mus_musculus] PASSED [ 53%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-homo_sapiens] PASSED [ 55%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-mus_musculus] PASSED [ 57%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-homo_sapiens] PASSED           [ 60%]
api/python/cell_census/tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-mus_musculus] PASSED           [ 62%]
api/python/cell_census/tests/test_directory.py::test_get_census_version_directory PASSED                               [ 64%]
api/python/cell_census/tests/test_directory.py::test_get_census_version_description_errors PASSED                      [ 66%]
api/python/cell_census/tests/test_directory.py::test_live_directory_contents PASSED                                    [ 68%]
api/python/cell_census/tests/test_get_anndata.py::test_get_anndata_value_filter PASSED                                 [ 71%]
api/python/cell_census/tests/test_get_anndata.py::test_get_anndata_coords PASSED                                       [ 73%]
api/python/cell_census/tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter PASSED             [ 75%]
api/python/cell_census/tests/test_get_helpers.py::test_get_experiment PASSED                                           [ 77%]
api/python/cell_census/tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens] PASSED                        [ 80%]
api/python/cell_census/tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus] PASSED                        [ 82%]
api/python/cell_census/tests/test_open.py::test_open_soma_latest PASSED                                                [ 84%]
api/python/cell_census/tests/test_open.py::test_open_soma_with_context PASSED                                          [ 86%]
api/python/cell_census/tests/test_open.py::test_open_soma_errors PASSED                                                [ 88%]
api/python/cell_census/tests/test_open.py::test_get_source_h5ad_uri PASSED                                             [ 91%]
api/python/cell_census/tests/test_open.py::test_get_source_h5ad_uri_errors PASSED                                      [ 93%]
api/python/cell_census/tests/test_open.py::test_download_source_h5ad PASSED                                            [ 95%]
api/python/cell_census/tests/test_open.py::test_download_source_h5ad_errors PASSED                                     [ 97%]
api/python/cell_census/tests/test_util.py::test_uri_join PASSED                                                        [100%]

===================================================== slowest durations ======================================================
5455.14s call     tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-homo_sapiens]
1388.18s call     tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-homo_sapiens]
400.45s call     tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config4-homo_sapiens]
183.85s call     tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-mus_musculus]
110.33s call     tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-mus_musculus]
63.52s call     tests/test_acceptance.py::test_get_anndata[First 1M cells-mus_musculus]
44.27s call     tests/test_acceptance.py::test_get_anndata[First 1M cells-homo_sapiens]
35.95s call     tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config5-homo_sapiens]
25.85s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens]
24.19s call     tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config4-mus_musculus]
22.38s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens]
13.23s call     tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config5-mus_musculus]
11.56s call     tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
9.32s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus]
9.31s call     tests/test_acceptance.py::test_get_anndata[First 100K cells-homo_sapiens]
8.39s call     tests/test_acceptance.py::test_incremental_read[homo_sapiens]
8.14s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens]
7.60s call     tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-homo_sapiens]
7.25s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus]
7.25s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens]
7.23s call     tests/test_acceptance.py::test_load_axes[homo_sapiens]
6.91s call     tests/test_acceptance.py::test_get_anndata[First 100K cells-mus_musculus]
6.25s setup    tests/test_open.py::test_download_source_h5ad
5.88s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus]
5.58s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus]
5.14s call     tests/test_directory.py::test_live_directory_contents
5.13s call     tests/test_acceptance.py::test_get_anndata[First 10K cells-homo_sapiens]
4.89s call     tests/test_open.py::test_get_source_h5ad_uri
4.59s call     tests/test_open.py::test_open_soma_latest
4.35s call     tests/test_acceptance.py::test_incremental_read[mus_musculus]
4.23s call     tests/test_get_anndata.py::test_get_anndata_value_filter
3.96s call     tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-mus_musculus]
3.66s call     tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens]
3.37s call     tests/test_acceptance.py::test_get_anndata[First 10K cells-mus_musculus]
2.97s call     tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus]
2.62s call     tests/test_get_anndata.py::test_get_anndata_coords
2.35s call     tests/test_open.py::test_download_source_h5ad
2.04s call     tests/test_acceptance.py::test_load_axes[mus_musculus]
1.94s setup    tests/test_get_anndata.py::test_get_anndata_coords
1.21s call     tests/test_open.py::test_get_source_h5ad_uri_errors
0.99s setup    tests/test_open.py::test_download_source_h5ad_errors
0.55s call     tests/test_get_helpers.py::test_get_experiment
0.51s call     tests/test_open.py::test_open_soma_with_context
0.25s setup    tests/test_get_anndata.py::test_get_anndata_value_filter
0.23s setup    tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
0.06s call     tests/test_directory.py::test_get_census_version_description_errors
0.04s setup    tests/test_directory.py::test_get_census_version_directory
0.02s call     tests/test_directory.py::test_get_census_version_directory
0.01s teardown tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-homo_sapiens]
0.01s setup    tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config6-mus_musculus]
0.01s teardown tests/test_acceptance.py::test_get_anndata[None-None-ctx_config7-homo_sapiens]

(84 durations < 0.005s hidden.  Use -vv to show these durations.)
============================================== 45 passed in 7924.13s (2:12:04) ===============================================
```
