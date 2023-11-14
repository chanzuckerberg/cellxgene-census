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

## Acceptance (expensive) tests

These tests are periodically run, and are not part of CI due to their overhead.

When run, please record the results in this file (below) and commit the change to git. Please include the following information:

- date
- config:
  - EC2 instance type and any system config (i.e., swap)
  - host and OS as reported by `uname -a`. **Please remove IP address**
  - Python & package versions and OS - suggest capturing the output of `tiledbsoma.show_package_versions()`
  - The Census version used for the test (i.e., the version aliased as `latest`). This can be easily captured using `cellxgene_census.get_census_version_description('latest')`
  - the cellxgene_census package version (ie., `cellxgene_census.__version__`)
- any run notes
- full output of: `pytest -v --durations=0 --experimental --expensive ./api/python/cellxgene_census/tests/`

### 2023-10-23

- Host: EC2 instance type: `r6id.32xlarge`, all nvme mounted as swap.
- Uname: Linux 6.2.0-1015-aws #15~22.04.1-Ubuntu SMP Fri Oct  6 21:37:24 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Python & census versions:

```python
>>> import cellxgene_census, tiledbsoma
>>> tiledbsoma.show_package_versions()
tiledbsoma.__version__        1.5.1
TileDB-Py tiledb.version()    (0, 23, 4)
TileDB core version           2.17.4
libtiledbsoma version()       libtiledb=2.17.4
python version                3.10.12.final.0
OS version                    Linux 6.2.0-1015-aws
```

**Pytest output:**

```text
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-7.4.3, pluggy-1.3.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/ssm-user/cellxgene-census/api/python/cellxgene_census
configfile: pyproject.toml
plugins: requests-mock-1.11.0
collecting ... collected 411 items

api/python/cellxgene_census/tests/test_acceptance.py::test_load_axes[homo_sapiens] PASSED [  0%]
api/python/cellxgene_census/tests/test_acceptance.py::test_load_axes[mus_musculus] PASSED [  0%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_obs[2-None-homo_sapiens] PASSED [  0%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_obs[2-None-mus_musculus] PASSED [  0%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_obs[None-ctx_config1-homo_sapiens] PASSED [  1%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_obs[None-ctx_config1-mus_musculus] PASSED [  1%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_var[2-None-homo_sapiens] PASSED [  1%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_var[2-None-mus_musculus] PASSED [  1%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_var[None-ctx_config1-homo_sapiens] PASSED [  2%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_var[None-ctx_config1-mus_musculus] PASSED [  2%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[2-None-coords0-homo_sapiens] PASSED [  2%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[2-None-coords0-mus_musculus] PASSED [  2%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[None-ctx_config1-coords1-homo_sapiens] PASSED [  3%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[None-ctx_config1-coords1-mus_musculus] PASSED [  3%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[None-ctx_config2-coords2-homo_sapiens] PASSED [  3%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[None-ctx_config2-coords2-mus_musculus] PASSED [  3%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[None-ctx_config3-coords3-homo_sapiens] PASSED [  4%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[None-ctx_config3-coords3-mus_musculus] PASSED [  4%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens] PASSED [  4%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus] PASSED [  4%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens] PASSED [  5%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus] PASSED [  5%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens] PASSED [  5%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus] PASSED [  5%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens] PASSED [  6%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus] PASSED [  6%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-homo_sapiens] PASSED [  6%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-mus_musculus] PASSED [  6%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 10K cells-homo_sapiens] PASSED [  7%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 10K cells-mus_musculus] PASSED [  7%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 100K cells-homo_sapiens] PASSED [  7%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 100K cells-mus_musculus] PASSED [  7%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 250K cells-homo_sapiens] PASSED [  8%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 250K cells-mus_musculus] PASSED [  8%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 500K cells-homo_sapiens] PASSED [  8%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 500K cells-mus_musculus] PASSED [  8%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 750K cells-homo_sapiens] PASSED [  9%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 750K cells-mus_musculus] PASSED [  9%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 1M cells-homo_sapiens] PASSED [  9%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 1M cells-mus_musculus] PASSED [  9%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[tissue_general=='brain'-obs_coords7-ctx_config7-homo_sapiens] PASSED [  9%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[tissue_general=='brain'-obs_coords7-ctx_config7-mus_musculus] PASSED [ 10%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-obs_coords8-ctx_config8-homo_sapiens] PASSED [ 10%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-obs_coords8-ctx_config8-mus_musculus] PASSED [ 10%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[is_primary_data==True-obs_coords9-ctx_config9-homo_sapiens] PASSED [ 10%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[is_primary_data==True-obs_coords9-ctx_config9-mus_musculus] PASSED [ 11%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[None-obs_coords10-ctx_config10-homo_sapiens] PASSED [ 11%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[None-obs_coords10-ctx_config10-mus_musculus] PASSED [ 11%]
api/python/cellxgene_census/tests/test_directory.py::test_get_census_version_directory PASSED [ 11%]
api/python/cellxgene_census/tests/test_directory.py::test_get_census_version_directory__lts_only PASSED [ 12%]
api/python/cellxgene_census/tests/test_directory.py::test_get_census_version_directory__exclude_lts PASSED [ 12%]
api/python/cellxgene_census/tests/test_directory.py::test_get_census_version_directory__include_retracted PASSED [ 12%]
api/python/cellxgene_census/tests/test_directory.py::test_get_census_version_directory__retraction_info PASSED [ 12%]
api/python/cellxgene_census/tests/test_directory.py::test_get_census_version_description_errors PASSED [ 13%]
api/python/cellxgene_census/tests/test_directory.py::test_get_census_mirrors_directory PASSED [ 13%]
api/python/cellxgene_census/tests/test_directory.py::test_live_directory_contents PASSED [ 13%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_value_filter PASSED [ 13%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_coords PASSED [ 14%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter PASSED [ 14%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_x_layer[raw] PASSED [ 14%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_x_layer[normalized] PASSED [ 14%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_two_layers[layers0] PASSED [ 15%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_two_layers[layers1] PASSED [ 15%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_wrong_layer_names PASSED [ 15%]
api/python/cellxgene_census/tests/test_get_helpers.py::test_get_experiment PASSED [ 15%]
api/python/cellxgene_census/tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens] PASSED [ 16%]
api/python/cellxgene_census/tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus] PASSED [ 16%]
api/python/cellxgene_census/tests/test_lts_compat.py::test_open[stable] PASSED [ 16%]
api/python/cellxgene_census/tests/test_lts_compat.py::test_read_dataframe[stable] PASSED [ 16%]
api/python/cellxgene_census/tests/test_lts_compat.py::test_read_arrays[stable] PASSED [ 17%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_stable PASSED [ 17%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_latest PASSED [ 17%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_with_context FAILED [ 17%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_invalid_args PASSED [ 18%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_errors PASSED [ 18%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_uses_correct_mirror PASSED [ 18%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_rejects_non_s3_mirror PASSED [ 18%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_works_if_no_relative_uri_specified PASSED [ 18%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_defaults_to_stable PASSED [ 19%]
api/python/cellxgene_census/tests/test_open.py::test_get_source_h5ad_uri PASSED [ 19%]
api/python/cellxgene_census/tests/test_open.py::test_get_source_h5ad_uri_errors PASSED [ 19%]
api/python/cellxgene_census/tests/test_open.py::test_download_source_h5ad PASSED [ 19%]
api/python/cellxgene_census/tests/test_open.py::test_download_source_h5ad_errors PASSED [ 20%]
api/python/cellxgene_census/tests/test_open.py::test_opening_census_without_anon_access_fails_with_bogus_creds PASSED [ 20%]
api/python/cellxgene_census/tests/test_open.py::test_can_open_with_anonymous_access PASSED [ 20%]
api/python/cellxgene_census/tests/test_util.py::test_uri_join PASSED     [ 20%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_non_batched[6-3-pytorch_x_value_gen-True] PASSED [ 21%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_non_batched[6-3-pytorch_x_value_gen-False] PASSED [ 21%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_batching__all_batches_full_size[6-3-pytorch_x_value_gen-True] PASSED [ 21%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_batching__all_batches_full_size[6-3-pytorch_x_value_gen-False] PASSED [ 21%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_unique_soma_joinids[obs_range0-3-pytorch_x_value_gen-True] PASSED [ 22%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_unique_soma_joinids[obs_range1-3-pytorch_x_value_gen-False] PASSED [ 22%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_batching__partial_final_batch_size[5-3-pytorch_x_value_gen-True] PASSED [ 22%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_batching__partial_final_batch_size[5-3-pytorch_x_value_gen-False] PASSED [ 22%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_batching__exactly_one_batch[3-3-pytorch_x_value_gen-True] PASSED [ 23%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_batching__exactly_one_batch[3-3-pytorch_x_value_gen-False] PASSED [ 23%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_batching__empty_query_result[6-3-pytorch_x_value_gen-True] PASSED [ 23%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_batching__empty_query_result[6-3-pytorch_x_value_gen-False] PASSED [ 23%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_sparse_output__non_batched[6-3-pytorch_x_value_gen-True] PASSED [ 24%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_sparse_output__non_batched[6-3-pytorch_x_value_gen-False] PASSED [ 24%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_sparse_output__batched[6-3-pytorch_x_value_gen-True] PASSED [ 24%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_sparse_output__batched[6-3-pytorch_x_value_gen-False] PASSED [ 24%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_batching__partial_soma_batches_are_concatenated[10-1-pytorch_x_value_gen-True] PASSED [ 25%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_batching__partial_soma_batches_are_concatenated[10-1-pytorch_x_value_gen-False] PASSED [ 25%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_encoders[3-3-pytorch_x_value_gen] PASSED [ 25%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_multiprocessing__returns_full_result[6-3-pytorch_x_value_gen] PASSED [ 25%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_distributed__returns_data_partition_for_rank[6-3-pytorch_x_value_gen] PASSED [ 26%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_distributed_and_multiprocessing__returns_data_partition_for_rank[12-3-pytorch_x_value_gen] PASSED [ 26%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__non_batched[3-3-pytorch_x_value_gen-True] PASSED [ 26%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__non_batched[3-3-pytorch_x_value_gen-False] PASSED [ 26%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__batched[6-3-pytorch_x_value_gen-True] PASSED [ 27%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__batched[6-3-pytorch_x_value_gen-False] PASSED [ 27%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test__X_tensor_dtype_matches_X_matrix[6-3-pytorch_x_value_gen-True] PASSED [ 27%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test__X_tensor_dtype_matches_X_matrix[6-3-pytorch_x_value_gen-False] PASSED [ 27%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test__pytorch_splitting[10-1-pytorch_x_value_gen] PASSED [ 27%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test__shuffle[16-1-pytorch_seq_x_value_gen] PASSED [ 28%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__multiprocess_sparse_matrix__fails SKIPPED [ 28%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__multiprocess_dense_matrix__ok SKIPPED [ 28%]
api/python/cellxgene_census/tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__unsupported_params__fails PASSED [ 28%]
api/python/cellxgene_census/tests/experimental/ml/huggingface/test_geneformer.py::test_GeneformerTokenizer[4] SKIPPED [ 29%]
api/python/cellxgene_census/tests/experimental/ml/huggingface/test_geneformer.py::test_GeneformerTokenizer[100000] SKIPPED [ 29%]
api/python/cellxgene_census/tests/experimental/ml/huggingface/test_geneformer.py::test_GeneformerTokenizer_docstring_example SKIPPED [ 29%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 29%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 30%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 30%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 30%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 30%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 31%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 31%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 31%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 31%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 32%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 32%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 32%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 32%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 33%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 33%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 33%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 33%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 34%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 34%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 34%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 34%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 35%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 35%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 35%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 35%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 36%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 36%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 36%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 36%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 36%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 37%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 37%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 37%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 37%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 38%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 38%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 38%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 38%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 39%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 39%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 39%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 39%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 40%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 40%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 40%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 40%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 41%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 41%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 41%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 41%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 42%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 42%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 42%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 42%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 43%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 43%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 43%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 43%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 44%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 44%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 44%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 44%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 45%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 45%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 45%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 45%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 45%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 46%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 46%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 46%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 46%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 47%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 47%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 47%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 47%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 48%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 48%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 48%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 48%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 49%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 49%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 49%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 49%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 50%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 50%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 50%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 50%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 51%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 51%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 51%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 51%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 52%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 52%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 52%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 52%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 53%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 53%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 53%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 53%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 54%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 54%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 54%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 54%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 54%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 55%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 55%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 55%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 55%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 56%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 56%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 56%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 56%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 57%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 57%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 57%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 57%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 58%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 58%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 58%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 58%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 59%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 59%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 59%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 59%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "liver"] PASSED [ 60%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"] PASSED [ 60%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]] PASSED [ 60%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"] PASSED [ 60%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[mus_musculus-Mus musculus-tissue_general == "liver" and is_primary_data == True-None-None] PASSED [ 61%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[mus_musculus-Mus musculus-is_primary_data == True and tissue_general == "heart"-dataset_id-None] PASSED [ 61%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[mus_musculus-Mus musculus-is_primary_data == True and tissue_general == "heart"-batch_key2-None] PASSED [ 61%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[mus_musculus-Mus musculus-is_primary_data == True-dataset_id-obs_coords3] PASSED [ 61%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[homo_sapiens-Homo sapiens-is_primary_data == True-dataset_id-obs_coords4] FAILED [ 62%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_error_cases PASSED [ 62%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_max_loess_jitter_error PASSED [ 62%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_user_defined_batch_key_func[None] PASSED [ 62%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_user_defined_batch_key_func[suspension_type] PASSED [ 63%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_user_defined_batch_key_func[batch_key2] PASSED [ 63%]
api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_hvg_user_defined_batch_key_func[batch_key3] PASSED [ 63%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-1-101] PASSED [ 63%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-1-53] PASSED [ 63%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-3-101] PASSED [ 64%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-3-53] PASSED [ 64%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-11-101] PASSED [ 64%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-11-53] PASSED [ 64%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-101-101] PASSED [ 65%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-101-53] PASSED [ 65%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-1-101] PASSED [ 65%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-1-53] PASSED [ 65%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-3-101] PASSED [ 66%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-3-53] PASSED [ 66%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-11-101] PASSED [ 66%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-11-53] PASSED [ 66%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-101-101] PASSED [ 67%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-101-53] PASSED [ 67%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-1-101] PASSED [ 67%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-1-53] PASSED [ 67%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-3-101] PASSED [ 68%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-3-53] PASSED [ 68%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-11-101] PASSED [ 68%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-11-53] PASSED [ 68%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-101-101] PASSED [ 69%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-101-53] PASSED [ 69%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-1-101] PASSED [ 69%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-1-53] PASSED [ 69%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-3-101] PASSED [ 70%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-3-53] PASSED [ 70%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-11-101] PASSED [ 70%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-11-53] PASSED [ 70%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-101-101] PASSED [ 71%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-101-53] PASSED [ 71%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-1-101] PASSED [ 71%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-1-53] PASSED [ 71%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-3-101] PASSED [ 72%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-3-53] PASSED [ 72%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-11-101] PASSED [ 72%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-11-53] PASSED [ 72%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-101-101] PASSED [ 72%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-101-53] PASSED [ 73%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-1-101] PASSED [ 73%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-1-53] PASSED [ 73%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-3-101] PASSED [ 73%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-3-53] PASSED [ 74%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-11-101] PASSED [ 74%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-11-53] PASSED [ 74%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-101-101] PASSED [ 74%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-101-53] PASSED [ 75%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-1-101] PASSED [ 75%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-1-53] PASSED [ 75%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-3-101] PASSED [ 75%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-3-53] PASSED [ 76%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-11-101] PASSED [ 76%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-11-53] PASSED [ 76%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-101-101] PASSED [ 76%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-101-53] PASSED [ 77%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-1-101] PASSED [ 77%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-1-53] PASSED [ 77%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-3-101] PASSED [ 77%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-3-53] PASSED [ 78%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-11-101] PASSED [ 78%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-11-53] PASSED [ 78%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-101-101] PASSED [ 78%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-101-53] PASSED [ 79%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-1-101] PASSED [ 79%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-1-53] PASSED [ 79%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-3-101] PASSED [ 79%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-3-53] PASSED [ 80%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-11-101] PASSED [ 80%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-11-53] PASSED [ 80%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-101-101] PASSED [ 80%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-101-53] PASSED [ 81%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-1-101] PASSED [ 81%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-1-53] PASSED [ 81%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-3-101] PASSED [ 81%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-3-53] PASSED [ 81%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-11-101] PASSED [ 82%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-11-53] PASSED [ 82%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-101-101] PASSED [ 82%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-101-53] PASSED [ 82%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-1-101] PASSED [ 83%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-1-53] PASSED [ 83%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-3-101] PASSED [ 83%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-3-53] PASSED [ 83%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-11-101] PASSED [ 84%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-11-53] PASSED [ 84%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-101-101] PASSED [ 84%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-101-53] PASSED [ 84%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-1-101] PASSED [ 85%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-1-53] PASSED [ 85%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-3-101] PASSED [ 85%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-3-53] PASSED [ 85%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-11-101] PASSED [ 86%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-11-53] PASSED [ 86%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-101-101] PASSED [ 86%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-101-53] PASSED [ 86%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar_nnz_only_batches_fails[1200-511] PASSED [ 87%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_meanvar_nnz_only_batches_fails[100001-57] PASSED [ 87%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_mean[1200-511-101] PASSED [ 87%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_mean[1200-511-53] PASSED [ 87%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_mean[100001-57-101] PASSED [ 88%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_mean[100001-57-53] PASSED [ 88%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[1200-511-1-101] PASSED [ 88%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[1200-511-1-53] PASSED [ 88%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[1200-511-3-101] PASSED [ 89%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[1200-511-3-53] PASSED [ 89%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[1200-511-11-101] PASSED [ 89%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[1200-511-11-53] PASSED [ 89%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[1200-511-101-101] PASSED [ 90%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[1200-511-101-53] PASSED [ 90%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[100001-57-1-101] PASSED [ 90%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[100001-57-1-53] PASSED [ 90%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[100001-57-3-101] PASSED [ 90%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[100001-57-3-53] PASSED [ 91%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[100001-57-11-101] PASSED [ 91%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[100001-57-11-53] PASSED [ 91%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[100001-57-101-101] PASSED [ 91%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_counts[100001-57-101-53] PASSED [ 92%]
api/python/cellxgene_census/tests/experimental/pp/test_online.py::test_mean_fails_no_variables_or_samples PASSED [ 92%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-True-True-0] PASSED [ 92%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-True-True-1] PASSED [ 92%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-True-False-0] PASSED [ 93%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-True-False-1] PASSED [ 93%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-False-True-0] PASSED [ 93%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-False-True-1] PASSED [ 93%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-True-True-0] PASSED [ 94%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-True-True-1] PASSED [ 94%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-True-False-0] PASSED [ 94%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-True-False-1] PASSED [ 94%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-False-True-0] PASSED [ 95%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-False-True-1] PASSED [ 95%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-True-True-0] PASSED [ 95%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-True-True-1] PASSED [ 95%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-True-False-0] PASSED [ 96%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-True-False-1] PASSED [ 96%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-False-True-0] PASSED [ 96%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-False-True-1] PASSED [ 96%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-True-0] FAILED [ 97%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-True-1] FAILED [ 97%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-False-0] FAILED [ 97%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-False-1] FAILED [ 97%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-False-True-0] FAILED [ 98%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-False-True-1] FAILED [ 98%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance_nnz_only[mus_musculus-obs_coords0-True-True-0] PASSED [ 98%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance_nnz_only[mus_musculus-obs_coords0-True-True-1] PASSED [ 98%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance_no_flags PASSED [ 99%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance_empty_query[mus_musculus] PASSED [ 99%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance_wrong_axis PASSED [ 99%]
api/python/cellxgene_census/tests/experimental/util/test_csr_iter.py::test_X_sparse_iter PASSED [ 99%]
api/python/cellxgene_census/tests/experimental/util/test_csr_iter.py::test_X_sparse_iter_unsupported PASSED [100%]

============================== slowest durations ===============================
336.88s call     tests/test_acceptance.py::test_get_anndata[None-obs_coords10-ctx_config10-homo_sapiens]
316.66s call     tests/test_acceptance.py::test_get_anndata[tissue_general=='brain'-obs_coords7-ctx_config7-homo_sapiens]
276.94s call     tests/test_acceptance.py::test_get_anndata[None-obs_coords10-ctx_config10-mus_musculus]
235.17s call     tests/test_acceptance.py::test_get_anndata[tissue_general=='brain'-obs_coords7-ctx_config7-mus_musculus]
235.12s call     tests/test_acceptance.py::test_get_anndata[is_primary_data==True-obs_coords9-ctx_config9-mus_musculus]
162.02s call     tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[mus_musculus-Mus musculus-is_primary_data == True-dataset_id-obs_coords3]
151.34s call     tests/test_acceptance.py::test_get_anndata[First 1M cells-homo_sapiens]
120.43s call     tests/test_acceptance.py::test_get_anndata[First 750K cells-homo_sapiens]
114.92s call     tests/test_acceptance.py::test_get_anndata[First 1M cells-mus_musculus]
83.01s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens]
82.79s call     tests/test_acceptance.py::test_get_anndata[First 750K cells-mus_musculus]
82.26s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config1-coords1-homo_sapiens]
76.06s call     tests/test_acceptance.py::test_get_anndata[First 500K cells-homo_sapiens]
73.13s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config2-coords2-homo_sapiens]
71.89s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
63.83s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
52.57s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
51.95s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
51.87s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
51.51s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
51.41s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
51.37s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
51.14s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
50.82s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
50.71s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
50.48s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
50.41s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
50.28s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
50.21s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
50.14s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
50.05s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
49.70s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
49.68s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
49.62s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
49.59s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
49.58s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
49.51s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
49.25s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
49.20s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
49.20s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
49.19s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
48.98s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
48.91s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
48.80s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
48.70s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
48.64s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
48.60s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
48.59s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
48.43s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config1-coords1-mus_musculus]
48.31s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
48.15s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
48.10s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
47.99s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
47.92s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
47.81s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-50-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
47.75s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
47.70s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
47.65s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-500-mus_musculus-is_primary_data == True and assay == "Smart-seq2"]
47.56s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
47.50s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
47.37s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
47.35s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
47.15s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
47.10s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
46.89s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
46.85s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
46.62s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
46.32s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
45.57s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
45.47s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
45.37s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
45.36s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
45.02s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
44.96s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
44.81s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
44.50s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
44.27s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
43.98s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
43.54s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general in ["heart", "lung"]]
43.32s call     tests/test_acceptance.py::test_get_anndata[First 500K cells-mus_musculus]
41.88s call     tests/test_acceptance.py::test_get_anndata[First 250K cells-homo_sapiens]
40.84s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens]
40.68s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config2-coords2-mus_musculus]
33.42s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-True-True-0]
31.24s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus]
26.90s call     tests/test_acceptance.py::test_get_anndata[is_primary_data==True-obs_coords9-ctx_config9-homo_sapiens]
26.41s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus]
23.43s call     tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[mus_musculus-Mus musculus-tissue_general == "liver" and is_primary_data == True-None-None]
23.29s call     tests/experimental/pp/test_hvg.py::test_hvg_user_defined_batch_key_func[None]
22.18s call     tests/test_acceptance.py::test_get_anndata[First 100K cells-homo_sapiens]
21.76s call     tests/experimental/util/test_csr_iter.py::test_X_sparse_iter
21.70s call     tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[mus_musculus-Mus musculus-is_primary_data == True and tissue_general == "heart"-dataset_id-None]
19.91s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
19.86s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
19.11s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-True-True-1]
18.78s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
18.71s call     tests/test_acceptance.py::test_get_anndata[First 250K cells-mus_musculus]
18.26s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
17.95s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.91s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.82s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
17.68s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.68s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.64s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.56s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.54s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.47s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.46s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
17.43s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
17.42s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.41s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-True-False-1]
17.41s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
17.40s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
17.39s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.36s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.31s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.28s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
17.17s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.17s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
17.16s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-None-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.13s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.11s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-False-True-1]
17.06s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
17.06s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.04s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.03s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
17.01s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
17.00s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-True-False-0]
17.00s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.98s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
16.93s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.91s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
16.89s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
16.89s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.85s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.70s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
16.69s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.67s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.62s call     tests/test_get_anndata.py::test_get_anndata_two_layers[layers0]
16.61s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.61s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
16.60s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.58s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.56s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
16.54s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
16.52s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.51s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.49s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.37s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-None-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.33s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-dataset_id-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.33s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
16.24s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True-obs_coords2-False-True-0]
16.15s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
16.06s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
15.99s call     tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[mus_musculus-Mus musculus-is_primary_data == True and tissue_general == "heart"-batch_key2-None]
15.97s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key3-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
15.94s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-True-True-0]
15.76s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-batch_key2-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
15.60s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
15.52s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-None-None-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
15.51s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general == "liver"]
15.49s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-True-True-0]
15.40s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[stable-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
15.31s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
15.13s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-500-mus_musculus-is_primary_data == True and tissue_general == "liver"]
14.94s call     tests/experimental/pp/test_hvg.py::test_hvg_vs_scanpy[latest-0.5-None-50-mus_musculus-is_primary_data == True and tissue_general == "skin of body"]
14.55s call     tests/test_get_anndata.py::test_get_anndata_two_layers[layers1]
13.72s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-True-False-0]
13.30s call     tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
13.27s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-True-True-1]
13.22s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-True-False-1]
13.20s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-False-True-1]
12.98s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-is_primary_data == True and tissue_general == "heart"-obs_coords1-False-True-0]
12.61s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config3-coords3-homo_sapiens]
12.55s call     tests/experimental/pp/test_hvg.py::test_max_loess_jitter_error
12.31s call     tests/experimental/pp/test_hvg.py::test_hvg_user_defined_batch_key_func[suspension_type]
11.96s call     tests/experimental/pp/test_hvg.py::test_hvg_user_defined_batch_key_func[batch_key2]
11.41s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus]
10.73s call     tests/experimental/pp/test_hvg.py::test_hvg_user_defined_batch_key_func[batch_key3]
10.66s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-False-True-1]
10.59s call     tests/test_acceptance.py::test_get_anndata[First 100K cells-mus_musculus]
10.59s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-True-True-1]
10.44s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-True-False-0]
10.41s call     tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-homo_sapiens]
10.18s call     tests/test_acceptance.py::test_get_anndata[First 10K cells-homo_sapiens]
10.07s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-True-False-1]
10.02s call     tests/experimental/pp/test_stats.py::test_mean_variance[mus_musculus-tissue_general == "liver" and is_primary_data == True-obs_coords0-False-True-0]
9.61s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens]
9.19s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus]
8.98s call     tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-obs_coords8-ctx_config8-mus_musculus]
8.74s call     tests/test_acceptance.py::test_load_axes[homo_sapiens]
8.62s call     tests/test_lts_compat.py::test_read_arrays[stable]
8.23s call     tests/test_get_anndata.py::test_get_anndata_x_layer[raw]
8.18s call     tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-mus_musculus]
8.10s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens]
7.92s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config3-coords3-mus_musculus]
7.90s call     tests/test_acceptance.py::test_incremental_read_X[2-None-coords0-homo_sapiens]
7.88s call     tests/test_get_anndata.py::test_get_anndata_x_layer[normalized]
7.37s call     tests/test_lts_compat.py::test_read_dataframe[stable]
6.98s call     tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-obs_coords8-ctx_config8-homo_sapiens]
6.05s call     tests/experimental/pp/test_stats.py::test_mean_variance_nnz_only[mus_musculus-obs_coords0-True-True-0]
5.92s call     tests/test_directory.py::test_live_directory_contents
5.90s call     tests/test_get_anndata.py::test_get_anndata_value_filter
5.72s call     tests/experimental/pp/test_stats.py::test_mean_variance_nnz_only[mus_musculus-obs_coords0-True-True-1]
5.20s call     tests/test_acceptance.py::test_get_anndata[First 10K cells-mus_musculus]
4.72s call     tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens]
4.55s call     tests/test_open.py::test_get_source_h5ad_uri
4.54s call     tests/experimental/ml/test_pytorch.py::test_multiprocessing__returns_full_result[6-3-pytorch_x_value_gen]
4.19s call     tests/test_acceptance.py::test_incremental_read_X[2-None-coords0-mus_musculus]
3.46s call     tests/test_acceptance.py::test_load_axes[mus_musculus]
3.27s call     tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[homo_sapiens-Homo sapiens-is_primary_data == True-dataset_id-obs_coords4]
3.25s call     tests/test_acceptance.py::test_incremental_read_obs[None-ctx_config1-homo_sapiens]
2.88s call     tests/test_get_anndata.py::test_get_anndata_coords
2.72s call     tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus]
2.50s call     tests/test_acceptance.py::test_incremental_read_obs[None-ctx_config1-mus_musculus]
2.44s call     tests/test_lts_compat.py::test_open[stable]
2.39s call     tests/experimental/ml/test_pytorch.py::test_batching__partial_soma_batches_are_concatenated[10-1-pytorch_x_value_gen-False]
2.04s call     tests/experimental/ml/test_pytorch.py::test_batching__partial_soma_batches_are_concatenated[10-1-pytorch_x_value_gen-True]
1.90s call     tests/test_acceptance.py::test_incremental_read_obs[2-None-homo_sapiens]
1.88s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-1-101]
1.85s call     tests/test_acceptance.py::test_incremental_read_obs[2-None-mus_musculus]
1.81s call     tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-True-0]
1.80s call     tests/test_directory.py::test_get_census_version_directory__include_retracted
1.80s call     tests/test_acceptance.py::test_incremental_read_var[2-None-homo_sapiens]
1.61s call     tests/test_open.py::test_download_source_h5ad
1.51s call     tests/test_acceptance.py::test_incremental_read_var[None-ctx_config1-homo_sapiens]
1.51s call     tests/test_acceptance.py::test_incremental_read_var[None-ctx_config1-mus_musculus]
1.42s call     tests/test_acceptance.py::test_incremental_read_var[2-None-mus_musculus]
1.40s call     tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-True-1]
1.40s call     tests/experimental/ml/test_pytorch.py::test_batching__partial_final_batch_size[5-3-pytorch_x_value_gen-False]
1.39s call     tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-False-0]
1.33s call     tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-False-True-1]
1.32s call     tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-False-1]
1.29s call     tests/experimental/pp/test_stats.py::test_mean_variance_empty_query[mus_musculus]
1.27s call     tests/experimental/ml/test_pytorch.py::test_batching__partial_final_batch_size[5-3-pytorch_x_value_gen-True]
1.21s call     tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-False-True-0]
1.16s setup    tests/test_open.py::test_download_source_h5ad
1.10s call     tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__batched[6-3-pytorch_x_value_gen-False]
1.07s call     tests/experimental/ml/test_pytorch.py::test_batching__exactly_one_batch[3-3-pytorch_x_value_gen-False]
1.07s setup    tests/test_open.py::test_download_source_h5ad_errors
1.06s call     tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__non_batched[3-3-pytorch_x_value_gen-False]
1.05s call     tests/experimental/ml/test_pytorch.py::test_batching__all_batches_full_size[6-3-pytorch_x_value_gen-False]
1.04s call     tests/experimental/ml/test_pytorch.py::test_unique_soma_joinids[obs_range1-3-pytorch_x_value_gen-False]
0.98s call     tests/test_open.py::test_get_source_h5ad_uri_errors
0.95s call     tests/test_open.py::test_open_soma_stable
0.95s call     tests/experimental/ml/test_pytorch.py::test_batching__exactly_one_batch[3-3-pytorch_x_value_gen-True]
0.95s call     tests/experimental/ml/test_pytorch.py::test_distributed__returns_data_partition_for_rank[6-3-pytorch_x_value_gen]
0.94s call     tests/experimental/ml/test_pytorch.py::test__pytorch_splitting[10-1-pytorch_x_value_gen]
0.93s call     tests/experimental/ml/test_pytorch.py::test_batching__all_batches_full_size[6-3-pytorch_x_value_gen-True]
0.93s call     tests/experimental/ml/test_pytorch.py::test_distributed_and_multiprocessing__returns_data_partition_for_rank[12-3-pytorch_x_value_gen]
0.93s call     tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__batched[6-3-pytorch_x_value_gen-True]
0.93s call     tests/experimental/ml/test_pytorch.py::test_unique_soma_joinids[obs_range0-3-pytorch_x_value_gen-True]
0.92s call     tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__non_batched[3-3-pytorch_x_value_gen-True]
0.90s call     tests/experimental/ml/test_pytorch.py::test__shuffle[16-1-pytorch_seq_x_value_gen]
0.87s call     tests/experimental/util/test_csr_iter.py::test_X_sparse_iter_unsupported
0.77s call     tests/experimental/pp/test_hvg.py::test_hvg_error_cases
0.72s call     tests/experimental/ml/test_pytorch.py::test_sparse_output__batched[6-3-pytorch_x_value_gen-False]
0.72s call     tests/experimental/ml/test_pytorch.py::test_non_batched[6-3-pytorch_x_value_gen-True]
0.71s call     tests/experimental/ml/test_pytorch.py::test_sparse_output__non_batched[6-3-pytorch_x_value_gen-False]
0.71s call     tests/experimental/ml/test_pytorch.py::test_non_batched[6-3-pytorch_x_value_gen-False]
0.68s call     tests/experimental/ml/test_pytorch.py::test__X_tensor_dtype_matches_X_matrix[6-3-pytorch_x_value_gen-False]
0.68s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-1-53]
0.67s call     tests/test_get_helpers.py::test_get_experiment
0.66s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-1-53]
0.62s call     tests/experimental/ml/test_pytorch.py::test_sparse_output__non_batched[6-3-pytorch_x_value_gen-True]
0.62s setup    tests/test_get_anndata.py::test_get_anndata_two_layers[layers0]
0.59s call     tests/experimental/ml/test_pytorch.py::test_encoders[3-3-pytorch_x_value_gen]
0.58s call     tests/experimental/ml/test_pytorch.py::test__X_tensor_dtype_matches_X_matrix[6-3-pytorch_x_value_gen-True]
0.58s call     tests/experimental/ml/test_pytorch.py::test_sparse_output__batched[6-3-pytorch_x_value_gen-True]
0.56s call     tests/experimental/ml/test_pytorch.py::test_batching__empty_query_result[6-3-pytorch_x_value_gen-False]
0.55s call     tests/experimental/ml/test_pytorch.py::test_batching__empty_query_result[6-3-pytorch_x_value_gen-True]
0.54s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-1-53]
0.54s call     tests/test_open.py::test_open_soma_with_context
0.54s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-1-101]
0.53s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-1-101]
0.53s call     tests/test_get_anndata.py::test_get_anndata_wrong_layer_names
0.50s setup    tests/test_get_anndata.py::test_get_anndata_x_layer[raw]
0.49s setup    tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
0.48s setup    tests/test_get_anndata.py::test_get_anndata_value_filter
0.48s setup    tests/test_get_anndata.py::test_get_anndata_two_layers[layers1]
0.48s setup    tests/test_get_anndata.py::test_get_anndata_x_layer[normalized]
0.46s call     tests/test_open.py::test_open_soma_latest
0.46s setup    tests/test_get_anndata.py::test_get_anndata_coords
0.45s setup    tests/test_get_anndata.py::test_get_anndata_wrong_layer_names
0.43s call     tests/test_open.py::test_can_open_with_anonymous_access
0.35s call     tests/test_open.py::test_opening_census_without_anon_access_fails_with_bogus_creds
0.24s setup    tests/experimental/ml/test_pytorch.py::test_non_batched[6-3-pytorch_x_value_gen-True]
0.22s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-1-101]
0.19s setup    tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__non_batched[3-3-pytorch_x_value_gen-False]
0.18s setup    tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__batched[6-3-pytorch_x_value_gen-False]
0.18s setup    tests/experimental/ml/test_pytorch.py::test__X_tensor_dtype_matches_X_matrix[6-3-pytorch_x_value_gen-True]
0.18s setup    tests/experimental/ml/test_pytorch.py::test_sparse_output__non_batched[6-3-pytorch_x_value_gen-True]
0.18s setup    tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__batched[6-3-pytorch_x_value_gen-True]
0.18s setup    tests/experimental/ml/test_pytorch.py::test_experiment_dataloader__non_batched[3-3-pytorch_x_value_gen-True]
0.18s setup    tests/experimental/ml/test_pytorch.py::test_batching__exactly_one_batch[3-3-pytorch_x_value_gen-True]
0.18s setup    tests/experimental/ml/test_pytorch.py::test_batching__exactly_one_batch[3-3-pytorch_x_value_gen-False]
0.18s setup    tests/experimental/ml/test_pytorch.py::test__pytorch_splitting[10-1-pytorch_x_value_gen]
0.18s setup    tests/experimental/ml/test_pytorch.py::test_distributed_and_multiprocessing__returns_data_partition_for_rank[12-3-pytorch_x_value_gen]
0.18s call     tests/test_directory.py::test_get_census_version_description_errors
0.18s setup    tests/experimental/ml/test_pytorch.py::test_batching__empty_query_result[6-3-pytorch_x_value_gen-True]
0.18s setup    tests/experimental/ml/test_pytorch.py::test_encoders[3-3-pytorch_x_value_gen]
0.18s setup    tests/experimental/ml/test_pytorch.py::test_batching__partial_soma_batches_are_concatenated[10-1-pytorch_x_value_gen-False]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_sparse_output__batched[6-3-pytorch_x_value_gen-True]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_batching__empty_query_result[6-3-pytorch_x_value_gen-False]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_batching__all_batches_full_size[6-3-pytorch_x_value_gen-True]
0.17s setup    tests/experimental/ml/test_pytorch.py::test__shuffle[16-1-pytorch_seq_x_value_gen]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_batching__partial_final_batch_size[5-3-pytorch_x_value_gen-False]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_sparse_output__batched[6-3-pytorch_x_value_gen-False]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_non_batched[6-3-pytorch_x_value_gen-False]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_unique_soma_joinids[obs_range0-3-pytorch_x_value_gen-True]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_sparse_output__non_batched[6-3-pytorch_x_value_gen-False]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_unique_soma_joinids[obs_range1-3-pytorch_x_value_gen-False]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_batching__partial_final_batch_size[5-3-pytorch_x_value_gen-True]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_distributed__returns_data_partition_for_rank[6-3-pytorch_x_value_gen]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_batching__all_batches_full_size[6-3-pytorch_x_value_gen-False]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_batching__partial_soma_batches_are_concatenated[10-1-pytorch_x_value_gen-True]
0.17s setup    tests/experimental/ml/test_pytorch.py::test__X_tensor_dtype_matches_X_matrix[6-3-pytorch_x_value_gen-False]
0.17s setup    tests/experimental/ml/test_pytorch.py::test_multiprocessing__returns_full_result[6-3-pytorch_x_value_gen]
0.16s call     tests/test_open.py::test_open_soma_errors
0.15s call     tests/test_open.py::test_open_soma_works_if_no_relative_uri_specified
0.15s call     tests/test_open.py::test_open_soma_defaults_to_stable
0.15s call     tests/test_open.py::test_open_soma_rejects_non_s3_mirror
0.14s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-101-53]
0.14s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-101-53]
0.14s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-101-53]
0.14s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-101-53]
0.14s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-101-53]
0.14s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-101-53]
0.13s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-101-101]
0.13s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-101-101]
0.13s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-101-101]
0.13s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-101-101]
0.13s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-101-101]
0.13s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-101-101]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-11-53]
0.11s call     tests/experimental/pp/test_online.py::test_counts[100001-57-101-53]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-11-53]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-11-53]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-3-53]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-11-53]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-11-53]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-11-53]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-3-53]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-3-101]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-3-53]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-3-53]
0.11s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-3-53]
0.10s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-11-101]
0.10s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-3-53]
0.10s call     tests/experimental/pp/test_online.py::test_counts[100001-57-101-101]
0.10s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-11-101]
0.10s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-11-101]
0.10s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-11-101]
0.10s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-11-101]
0.10s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-11-101]
0.10s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-3-101]
0.09s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-3-101]
0.09s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-3-101]
0.09s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-3-101]
0.09s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-3-101]
0.09s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-1-53]
0.09s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-1-53]
0.09s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-1-53]
0.09s call     tests/experimental/pp/test_online.py::test_counts[100001-57-3-53]
0.09s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-1-101]
0.09s call     tests/experimental/pp/test_online.py::test_counts[100001-57-11-53]
0.08s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-1-101]
0.08s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-1-101]
0.08s call     tests/experimental/pp/test_online.py::test_counts[100001-57-11-101]
0.07s call     tests/experimental/pp/test_online.py::test_counts[100001-57-1-53]
0.07s call     tests/experimental/pp/test_online.py::test_counts[100001-57-3-101]
0.06s call     tests/experimental/pp/test_online.py::test_counts[100001-57-1-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-1-101]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-1-101]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-1-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-1-101]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-1-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-1-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-101-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-101-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-101-101]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_counts[100001-57-1-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-1-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-101-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-101-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-101-53]
0.03s setup    tests/experimental/pp/test_online.py::test_counts[100001-57-3-53]
0.03s setup    tests/experimental/pp/test_online.py::test_counts[100001-57-11-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_counts[100001-57-101-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-101-101]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-101-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-3-53]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-1-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-3-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-1-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-11-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-3-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-11-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-3-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-11-101]
0.03s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-1-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-3-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-3-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-11-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-11-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-11-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-1-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-11-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-3-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-1-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-1-53]
0.03s setup    tests/experimental/pp/test_online.py::test_counts[100001-57-1-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-3-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-1-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-11-53]
0.03s setup    tests/experimental/pp/test_online.py::test_counts[100001-57-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-11-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-11-53]
0.03s setup    tests/experimental/pp/test_online.py::test_counts[100001-57-11-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-11-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-11-53]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-101-101]
0.03s setup    tests/experimental/pp/test_online.py::test_mean[100001-57-53]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[True-0-100001-57-101-53]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-101-53]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-101-53]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[True-1-100001-57-101-53]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[True-100-100001-57-101-53]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-101-53]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar_nnz_only_batches_fails[100001-57]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-3-101]
0.02s setup    tests/experimental/pp/test_online.py::test_mean[100001-57-101]
0.02s setup    tests/experimental/pp/test_online.py::test_counts[100001-57-3-101]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-3-101]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[False-1-100001-57-1-53]
0.02s call     tests/experimental/pp/test_online.py::test_counts[1200-511-101-53]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[False-0-100001-57-1-53]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-3-101]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-1-53]
0.02s call     tests/experimental/pp/test_online.py::test_counts[1200-511-101-101]
0.02s setup    tests/experimental/pp/test_online.py::test_meanvar[False-100-100001-57-3-53]
0.02s call     tests/experimental/pp/test_online.py::test_mean[100001-57-53]
0.01s call     tests/experimental/pp/test_online.py::test_mean[100001-57-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-3-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-3-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-11-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-11-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-11-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-11-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-11-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-11-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-3-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-3-101]
0.01s call     tests/experimental/pp/test_online.py::test_counts[1200-511-11-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-11-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-11-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-11-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-11-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-3-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-11-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-3-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-11-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-3-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-100-1200-511-3-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-3-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[True-1-1200-511-3-101]
0.01s call     tests/experimental/pp/test_online.py::test_counts[1200-511-11-101]
0.01s call     tests/experimental/pp/test_online.py::test_counts[1200-511-3-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-3-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-3-101]
0.01s call     tests/experimental/pp/test_online.py::test_counts[1200-511-3-101]
0.01s call     tests/experimental/pp/test_online.py::test_counts[1200-511-1-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-1-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-1-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-1-1200-511-1-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-0-1200-511-1-101]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-1-53]
0.01s call     tests/experimental/pp/test_online.py::test_meanvar[False-100-1200-511-1-101]
0.01s call     tests/experimental/pp/test_online.py::test_counts[1200-511-1-101]
0.01s setup    tests/experimental/pp/test_online.py::test_meanvar[True-0-1200-511-1-101]

(739 durations < 0.005s hidden.  Use -vv to show these durations.)
=========================== short test summary info ============================
FAILED api/python/cellxgene_census/tests/test_open.py::test_open_soma_with_context
FAILED api/python/cellxgene_census/tests/experimental/pp/test_hvg.py::test_get_highly_variable_genes[homo_sapiens-Homo sapiens-is_primary_data == True-dataset_id-obs_coords4]
FAILED api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-True-0]
FAILED api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-True-1]
FAILED api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-False-0]
FAILED api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-True-False-1]
FAILED api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-False-True-0]
FAILED api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance[homo_sapiens-is_primary_data == True-obs_coords3-False-True-1]
====== 8 failed, 398 passed, 5 skipped, 95 warnings in 7698.98s (2:08:18) ======
```

Note: failures have been identified to be related to the tests themselves, so we won't block the LTS release.

### 2023-07-26

- Host: EC2 instance type: `r6id.32xlarge`, all nvme mounted as swap.
- Uname: Linux 5.19.0-1028-aws #29~22.04.1-Ubuntu SMP Tue Jun 20 19:12:11 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Python & census versions:

```python
>>> import cellxgene_census, tiledbsoma
>>> tiledbsoma.show_package_versions()
tiledbsoma.__version__        1.2.7
TileDB-Py tiledb.version()    (0, 21, 3)
TileDB core version           2.15.2
libtiledbsoma version()       libtiledb=2.15.2
python version                3.10.6.final.0
OS version                    Linux 5.19.0-1028-aws
```

**Pytest output:**

```text
============================= test session starts ==============================
platform linux -- Python 3.10.6, pytest-7.1.3, pluggy-1.0.0 -- /home/ubuntu/venv/bin/python
cachedir: .pytest_cache
rootdir: /home/ubuntu/repos/cellxgene-census/api/python/cellxgene_census, configfile: pyproject.toml
plugins: anyio-3.6.2, requests-mock-1.11.0
collecting ... collected 274 items / 202 deselected / 72 selected

api/python/cellxgene_census/tests/test_acceptance.py::test_load_axes[homo_sapiens] PASSED [  1%]
api/python/cellxgene_census/tests/test_acceptance.py::test_load_axes[mus_musculus] PASSED [  2%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_obs[2-None-homo_sapiens] PASSED [  4%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_obs[2-None-mus_musculus] PASSED [  5%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_obs[None-ctx_config1-homo_sapiens] PASSED [  6%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_obs[None-ctx_config1-mus_musculus] PASSED [  8%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_var[2-None-homo_sapiens] PASSED [  9%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_var[2-None-mus_musculus] PASSED [ 11%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_var[None-ctx_config1-homo_sapiens] PASSED [ 12%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_var[None-ctx_config1-mus_musculus] PASSED [ 13%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[2-None-homo_sapiens] PASSED [ 15%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[2-None-mus_musculus] PASSED [ 16%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[None-ctx_config1-homo_sapiens] PASSED [ 18%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[None-ctx_config1-mus_musculus] PASSED [ 19%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[None-ctx_config2-homo_sapiens] PASSED [ 20%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_read_X[None-ctx_config2-mus_musculus] PASSED [ 22%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens] PASSED [ 23%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus] PASSED [ 25%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens] PASSED [ 26%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus] PASSED [ 27%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens] PASSED [ 29%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus] PASSED [ 30%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens] PASSED [ 31%]
api/python/cellxgene_census/tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus] PASSED [ 33%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-homo_sapiens] PASSED [ 34%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-mus_musculus] PASSED [ 36%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 10K cells-homo_sapiens] PASSED [ 37%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 10K cells-mus_musculus] PASSED [ 38%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 100K cells-homo_sapiens] PASSED [ 40%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 100K cells-mus_musculus] PASSED [ 41%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 250K cells-homo_sapiens] PASSED [ 43%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 250K cells-mus_musculus] PASSED [ 44%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 500K cells-homo_sapiens] PASSED [ 45%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 500K cells-mus_musculus] PASSED [ 47%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 750K cells-homo_sapiens] PASSED [ 48%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 750K cells-mus_musculus] PASSED [ 50%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 1M cells-homo_sapiens] PASSED [ 51%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[First 1M cells-mus_musculus] PASSED [ 52%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config7-homo_sapiens] PASSED [ 54%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config7-mus_musculus] PASSED [ 55%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-homo_sapiens] PASSED [ 56%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-mus_musculus] PASSED [ 58%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-homo_sapiens] PASSED [ 59%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-mus_musculus] PASSED [ 61%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[None-None-ctx_config10-homo_sapiens] PASSED [ 62%]
api/python/cellxgene_census/tests/test_acceptance.py::test_get_anndata[None-None-ctx_config10-mus_musculus] PASSED [ 63%]
api/python/cellxgene_census/tests/test_directory.py::test_get_census_version_directory PASSED [ 65%]
api/python/cellxgene_census/tests/test_directory.py::test_get_census_version_description_errors PASSED [ 66%]
api/python/cellxgene_census/tests/test_directory.py::test_live_directory_contents PASSED [ 68%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_value_filter PASSED [ 69%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_coords PASSED [ 70%]
api/python/cellxgene_census/tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter PASSED [ 72%]
api/python/cellxgene_census/tests/test_get_helpers.py::test_get_experiment PASSED [ 73%]
api/python/cellxgene_census/tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens] PASSED [ 75%]
api/python/cellxgene_census/tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus] PASSED [ 76%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_stable PASSED [ 77%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_latest PASSED [ 79%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_with_context PASSED [ 80%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_invalid_args PASSED [ 81%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_errors PASSED [ 83%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_defaults_to_latest_if_missing_stable PASSED [ 84%]
api/python/cellxgene_census/tests/test_open.py::test_open_soma_defaults_to_stable PASSED [ 86%]
api/python/cellxgene_census/tests/test_open.py::test_get_source_h5ad_uri PASSED [ 87%]
api/python/cellxgene_census/tests/test_open.py::test_get_source_h5ad_uri_errors PASSED [ 88%]
api/python/cellxgene_census/tests/test_open.py::test_download_source_h5ad PASSED [ 90%]
api/python/cellxgene_census/tests/test_open.py::test_download_source_h5ad_errors PASSED [ 91%]
api/python/cellxgene_census/tests/test_open.py::test_opening_census_without_anon_access_fails_with_bogus_creds PASSED [ 93%]
api/python/cellxgene_census/tests/test_open.py::test_can_open_with_anonymous_access PASSED [ 94%]
api/python/cellxgene_census/tests/test_util.py::test_uri_join PASSED     [ 95%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance_no_flags PASSED [ 97%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance_empty_query[mus_musculus] PASSED [ 98%]
api/python/cellxgene_census/tests/experimental/pp/test_stats.py::test_mean_variance_wrong_axis PASSED [100%]

============================== slowest durations ===============================
8283.70s call     tests/test_acceptance.py::test_get_anndata[None-None-ctx_config10-homo_sapiens]
1767.98s call     tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-homo_sapiens]
1304.89s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config1-homo_sapiens]
953.35s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config2-homo_sapiens]
903.42s call     tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-homo_sapiens]
309.04s call     tests/test_acceptance.py::test_get_anndata[None-None-ctx_config10-mus_musculus]
195.72s call     tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-mus_musculus]
135.65s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config1-mus_musculus]
115.40s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config2-mus_musculus]
65.46s call     tests/test_acceptance.py::test_get_anndata[First 1M cells-mus_musculus]
53.50s call     tests/test_acceptance.py::test_get_anndata[First 750K cells-mus_musculus]
44.52s call     tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-mus_musculus]
38.62s call     tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config7-homo_sapiens]
38.56s call     tests/test_acceptance.py::test_get_anndata[First 500K cells-mus_musculus]
34.90s call     tests/test_acceptance.py::test_get_anndata[First 1M cells-homo_sapiens]
31.58s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens]
26.10s call     tests/test_acceptance.py::test_get_anndata[First 750K cells-homo_sapiens]
23.89s call     tests/test_acceptance.py::test_get_anndata[First 500K cells-homo_sapiens]
23.75s call     tests/test_acceptance.py::test_get_anndata[First 250K cells-mus_musculus]
19.00s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens]
16.80s call     tests/test_acceptance.py::test_get_anndata[First 250K cells-homo_sapiens]
12.47s call     tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config7-mus_musculus]
12.25s call     tests/test_acceptance.py::test_get_anndata[First 100K cells-mus_musculus]
10.88s call     tests/test_acceptance.py::test_get_anndata[First 100K cells-homo_sapiens]
10.52s call     tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
9.96s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens]
9.67s call     tests/test_acceptance.py::test_incremental_read_X[2-None-homo_sapiens]
9.45s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus]
9.11s call     tests/test_acceptance.py::test_load_axes[homo_sapiens]
8.81s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens]
7.96s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus]
7.77s call     tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-homo_sapiens]
7.14s call     tests/test_directory.py::test_live_directory_contents
5.47s call     tests/test_get_anndata.py::test_get_anndata_value_filter
5.30s call     tests/test_acceptance.py::test_get_anndata[First 10K cells-homo_sapiens]
4.82s call     tests/test_acceptance.py::test_get_anndata[First 10K cells-mus_musculus]
4.69s call     tests/test_open.py::test_get_source_h5ad_uri
4.47s call     tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens]
4.46s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus]
4.26s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus]
3.89s call     tests/test_acceptance.py::test_incremental_read_X[2-None-mus_musculus]
3.80s call     tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-mus_musculus]
3.39s call     tests/test_acceptance.py::test_incremental_read_obs[None-ctx_config1-homo_sapiens]
3.24s call     tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus]
3.02s call     tests/test_get_anndata.py::test_get_anndata_coords
2.94s call     tests/test_open.py::test_download_source_h5ad
2.54s call     tests/test_acceptance.py::test_load_axes[mus_musculus]
1.80s call     tests/test_acceptance.py::test_incremental_read_obs[2-None-homo_sapiens]
1.62s call     tests/test_acceptance.py::test_incremental_read_obs[2-None-mus_musculus]
1.54s call     tests/test_acceptance.py::test_incremental_read_obs[None-ctx_config1-mus_musculus]
1.40s call     tests/experimental/pp/test_stats.py::test_mean_variance_empty_query[mus_musculus]
1.37s call     tests/test_acceptance.py::test_incremental_read_var[2-None-mus_musculus]
1.34s call     tests/test_acceptance.py::test_incremental_read_var[None-ctx_config1-homo_sapiens]
1.34s call     tests/test_acceptance.py::test_incremental_read_var[2-None-homo_sapiens]
1.33s setup    tests/test_open.py::test_download_source_h5ad_errors
1.28s call     tests/test_acceptance.py::test_incremental_read_var[None-ctx_config1-mus_musculus]
1.20s setup    tests/test_open.py::test_download_source_h5ad
1.13s call     tests/test_open.py::test_get_source_h5ad_uri_errors
0.76s call     tests/test_open.py::test_open_soma_with_context
0.71s call     tests/test_open.py::test_open_soma_stable
0.56s call     tests/test_directory.py::test_get_census_version_description_errors
0.48s call     tests/test_get_helpers.py::test_get_experiment
0.47s setup    tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
0.43s setup    tests/test_get_anndata.py::test_get_anndata_coords
0.39s call     tests/test_open.py::test_open_soma_defaults_to_latest_if_missing_stable
0.34s setup    tests/test_get_anndata.py::test_get_anndata_value_filter
0.34s call     tests/test_open.py::test_open_soma_latest
0.33s call     tests/test_open.py::test_can_open_with_anonymous_access
0.26s call     tests/test_open.py::test_opening_census_without_anon_access_fails_with_bogus_creds
0.04s setup    tests/test_directory.py::test_get_census_version_directory
0.03s teardown tests/test_acceptance.py::test_get_anndata[None-None-ctx_config10-homo_sapiens]
0.03s setup    tests/test_acceptance.py::test_get_anndata[None-None-ctx_config10-mus_musculus]
0.02s teardown tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-homo_sapiens]
0.02s call     tests/test_directory.py::test_get_census_version_directory
0.02s setup    tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-mus_musculus]
0.01s setup    tests/experimental/pp/test_stats.py::test_mean_variance_empty_query[mus_musculus]
0.01s teardown tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-homo_sapiens]
0.01s setup    tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-mus_musculus]
0.01s call     tests/experimental/pp/test_stats.py::test_mean_variance_no_flags

(137 durations < 0.005s hidden.  Use -vv to show these durations.)
=============== 72 passed, 202 deselected in 14584.03s (4:03:04) ===============
```

### 2023-06-23

- Host: EC2 instance type: `r6id.32xlarge`, all nvme mounted as swap.
- Uname: Linux 5.19.0-1025-aws #26~22.04.1-Ubuntu SMP Mon Apr 24 01:58:15 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Python & census versions:

```python
>>> import cellxgene_census, tiledbsoma
>>> tiledbsoma.show_package_versions()
tiledbsoma.__version__        1.2.5
TileDB-Py tiledb.version()    (0, 21, 5)
TileDB core version           2.15.4
libtiledbsoma version()       libtiledb=2.15.2
python version                3.10.6.final.0
OS version
>>> cellxgene_census.__version__
'1.2.1'
>>> cellxgene_census.get_census_version_description('latest')
{'release_date': None, 'release_build': '2023-06-20', 'soma': {'uri': 's3://cellxgene-data-public/cell-census/2023-06-20/soma/', 's3_region': 'us-west-2'}, 'h5ads': {'uri': 's3://cellxgene-data-public/cell-census/2023-06-20/h5ads/', 's3_region': 'us-west-2'}, 'alias': 'latest'}
```

**Pytest output:**

```text
============================= test session starts ==============================
platform linux -- Python 3.10.6, pytest-7.4.0, pluggy-1.2.0 -- /home/ubuntu/venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/ubuntu/repos/cellxgene-census/api/python/cellxgene_census
configfile: pyproject.toml
plugins: requests-mock-1.11.0
collecting ... collected 180 items / 111 deselected / 69 selected

test_acceptance.py::test_load_axes[homo_sapiens] PASSED                  [  1%]
test_acceptance.py::test_load_axes[mus_musculus] PASSED                  [  2%]
test_acceptance.py::test_incremental_read_obs[2-None-homo_sapiens] PASSED [  4%]
test_acceptance.py::test_incremental_read_obs[2-None-mus_musculus] PASSED [  5%]
test_acceptance.py::test_incremental_read_obs[None-ctx_config1-homo_sapiens] PASSED [  7%]
test_acceptance.py::test_incremental_read_obs[None-ctx_config1-mus_musculus] PASSED [  8%]
test_acceptance.py::test_incremental_read_var[2-None-homo_sapiens] PASSED [ 10%]
test_acceptance.py::test_incremental_read_var[2-None-mus_musculus] PASSED [ 11%]
test_acceptance.py::test_incremental_read_var[None-ctx_config1-homo_sapiens] PASSED [ 13%]
test_acceptance.py::test_incremental_read_var[None-ctx_config1-mus_musculus] PASSED [ 14%]
test_acceptance.py::test_incremental_read_X[2-None-homo_sapiens] PASSED  [ 15%]
test_acceptance.py::test_incremental_read_X[2-None-mus_musculus] PASSED  [ 17%]
test_acceptance.py::test_incremental_read_X[None-ctx_config1-homo_sapiens] PASSED [ 18%]
test_acceptance.py::test_incremental_read_X[None-ctx_config1-mus_musculus] PASSED [ 20%]
test_acceptance.py::test_incremental_read_X[None-ctx_config2-homo_sapiens] PASSED [ 21%]
test_acceptance.py::test_incremental_read_X[None-ctx_config2-mus_musculus] PASSED [ 23%]
test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens] PASSED [ 24%]
test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus] PASSED [ 26%]
test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens] PASSED [ 27%]
test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus] PASSED [ 28%]
test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens] PASSED [ 30%]
test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus] PASSED [ 31%]
test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens] PASSED [ 33%]
test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus] PASSED [ 34%]
test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-homo_sapiens] PASSED [ 36%]
test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-mus_musculus] PASSED [ 37%]
test_acceptance.py::test_get_anndata[First 10K cells-homo_sapiens] PASSED [ 39%]
test_acceptance.py::test_get_anndata[First 10K cells-mus_musculus] PASSED [ 40%]
test_acceptance.py::test_get_anndata[First 100K cells-homo_sapiens] PASSED [ 42%]
test_acceptance.py::test_get_anndata[First 100K cells-mus_musculus] PASSED [ 43%]
test_acceptance.py::test_get_anndata[First 250K cells-homo_sapiens] PASSED [ 44%]
test_acceptance.py::test_get_anndata[First 250K cells-mus_musculus] PASSED [ 46%]
test_acceptance.py::test_get_anndata[First 500K cells-homo_sapiens] PASSED [ 47%]
test_acceptance.py::test_get_anndata[First 500K cells-mus_musculus] PASSED [ 49%]
test_acceptance.py::test_get_anndata[First 750K cells-homo_sapiens] PASSED [ 50%]
test_acceptance.py::test_get_anndata[First 750K cells-mus_musculus] PASSED [ 52%]
test_acceptance.py::test_get_anndata[First 1M cells-homo_sapiens] PASSED [ 53%]
test_acceptance.py::test_get_anndata[First 1M cells-mus_musculus] PASSED [ 55%]
test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config7-homo_sapiens] PASSED [ 56%]
test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config7-mus_musculus] PASSED [ 57%]
test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-homo_sapiens] PASSED [ 59%]
test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-mus_musculus] PASSED [ 60%]
test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-homo_sapiens] PASSED [ 62%]
test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-mus_musculus] PASSED [ 63%]
test_acceptance.py::test_get_anndata[None-None-ctx_config10-homo_sapiens] PASSED [ 65%]
test_acceptance.py::test_get_anndata[None-None-ctx_config10-mus_musculus] PASSED [ 66%]
test_directory.py::test_get_census_version_directory PASSED              [ 68%]
test_directory.py::test_get_census_version_description_errors PASSED     [ 69%]
test_directory.py::test_live_directory_contents PASSED                   [ 71%]
test_get_anndata.py::test_get_anndata_value_filter PASSED                [ 72%]
test_get_anndata.py::test_get_anndata_coords PASSED                      [ 73%]
test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter PASSED [ 75%]
test_get_helpers.py::test_get_experiment PASSED                          [ 76%]
test_get_helpers.py::test_get_presence_matrix[homo_sapiens] PASSED       [ 78%]
test_get_helpers.py::test_get_presence_matrix[mus_musculus] PASSED       [ 79%]
test_open.py::test_open_soma_stable PASSED                               [ 81%]
test_open.py::test_open_soma_latest PASSED                               [ 82%]
test_open.py::test_open_soma_with_context PASSED                         [ 84%]
test_open.py::test_open_soma_invalid_args PASSED                         [ 85%]
test_open.py::test_open_soma_errors PASSED                               [ 86%]
test_open.py::test_open_soma_defaults_to_latest_if_missing_stable PASSED [ 88%]
test_open.py::test_open_soma_defaults_to_stable PASSED                   [ 89%]
test_open.py::test_get_source_h5ad_uri PASSED                            [ 91%]
test_open.py::test_get_source_h5ad_uri_errors PASSED                     [ 92%]
test_open.py::test_download_source_h5ad PASSED                           [ 94%]
test_open.py::test_download_source_h5ad_errors PASSED                    [ 95%]
test_open.py::test_opening_census_without_anon_access_fails_with_bogus_creds PASSED [ 97%]
test_open.py::test_can_open_with_anonymous_access PASSED                 [ 98%]
test_util.py::test_uri_join PASSED                                       [100%]

============================== slowest durations ===============================
8331.89s call     tests/test_acceptance.py::test_get_anndata[None-None-ctx_config10-homo_sapiens]
2755.15s call     tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-homo_sapiens]
1146.66s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config1-homo_sapiens]
891.72s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config2-homo_sapiens]
650.78s call     tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-homo_sapiens]
287.95s call     tests/test_acceptance.py::test_get_anndata[None-None-ctx_config10-mus_musculus]
190.27s call     tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-mus_musculus]
118.72s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config1-mus_musculus]
102.51s call     tests/test_acceptance.py::test_incremental_read_X[None-ctx_config2-mus_musculus]
55.81s call     tests/test_acceptance.py::test_get_anndata[First 1M cells-mus_musculus]
48.63s call     tests/test_acceptance.py::test_get_anndata[First 750K cells-mus_musculus]
43.77s call     tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-mus_musculus]
36.64s call     tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config7-homo_sapiens]
35.32s call     tests/test_acceptance.py::test_get_anndata[First 1M cells-homo_sapiens]
34.39s call     tests/test_acceptance.py::test_get_anndata[First 500K cells-mus_musculus]
29.83s call     tests/test_acceptance.py::test_get_anndata[First 750K cells-homo_sapiens]
27.46s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-homo_sapiens]
23.21s call     tests/test_acceptance.py::test_get_anndata[First 500K cells-homo_sapiens]
20.32s call     tests/test_acceptance.py::test_get_anndata[First 250K cells-mus_musculus]
18.22s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-homo_sapiens]
16.58s call     tests/test_acceptance.py::test_get_anndata[First 250K cells-homo_sapiens]
11.48s call     tests/test_acceptance.py::test_get_anndata[tissue=='brain'-None-ctx_config7-mus_musculus]
10.18s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-homo_sapiens]
9.88s call     tests/test_acceptance.py::test_get_anndata[First 100K cells-homo_sapiens]
9.47s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='brain'-mus_musculus]
9.44s call     tests/test_acceptance.py::test_get_anndata[First 100K cells-mus_musculus]
8.86s call     tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
8.70s call     tests/test_acceptance.py::test_incremental_read_X[2-None-homo_sapiens]
7.92s call     tests/test_acceptance.py::test_load_axes[homo_sapiens]
7.32s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='brain'-mus_musculus]
7.22s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-homo_sapiens]
6.99s call     tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-homo_sapiens]
6.20s call     tests/test_directory.py::test_live_directory_contents
5.57s call     tests/test_get_anndata.py::test_get_anndata_value_filter
4.58s call     tests/test_acceptance.py::test_incremental_query[2-tissue=='aorta'-mus_musculus]
4.53s call     tests/test_acceptance.py::test_get_anndata[First 10K cells-homo_sapiens]
4.19s call     tests/test_open.py::test_get_source_h5ad_uri
4.03s call     tests/test_get_helpers.py::test_get_presence_matrix[homo_sapiens]
4.01s call     tests/test_acceptance.py::test_incremental_query[None-tissue=='aorta'-mus_musculus]
3.98s call     tests/test_acceptance.py::test_get_anndata[First 10K cells-mus_musculus]
3.48s call     tests/test_acceptance.py::test_get_anndata[tissue=='aorta'-None-ctx_config0-mus_musculus]
3.41s call     tests/test_acceptance.py::test_incremental_read_obs[None-ctx_config1-homo_sapiens]
2.24s call     tests/test_get_anndata.py::test_get_anndata_coords
2.19s call     tests/test_acceptance.py::test_incremental_read_X[2-None-mus_musculus]
2.17s call     tests/test_get_helpers.py::test_get_presence_matrix[mus_musculus]
2.04s call     tests/test_acceptance.py::test_load_axes[mus_musculus]
2.01s call     tests/test_open.py::test_download_source_h5ad
1.38s call     tests/test_acceptance.py::test_incremental_read_obs[2-None-homo_sapiens]
1.25s call     tests/test_acceptance.py::test_incremental_read_obs[None-ctx_config1-mus_musculus]
1.24s call     tests/test_acceptance.py::test_incremental_read_obs[2-None-mus_musculus]
1.16s call     tests/test_acceptance.py::test_incremental_read_var[2-None-mus_musculus]
1.15s call     tests/test_acceptance.py::test_incremental_read_var[None-ctx_config1-mus_musculus]
1.12s call     tests/test_acceptance.py::test_incremental_read_var[2-None-homo_sapiens]
1.08s call     tests/test_acceptance.py::test_incremental_read_var[None-ctx_config1-homo_sapiens]
1.01s setup    tests/test_open.py::test_download_source_h5ad_errors
1.01s setup    tests/test_open.py::test_download_source_h5ad
0.90s call     tests/test_open.py::test_get_source_h5ad_uri_errors
0.68s call     tests/test_open.py::test_open_soma_stable
0.64s call     tests/test_open.py::test_open_soma_with_context
0.56s call     tests/test_get_helpers.py::test_get_experiment
0.36s call     tests/test_open.py::test_open_soma_defaults_to_latest_if_missing_stable
0.36s setup    tests/test_get_anndata.py::test_get_anndata_allows_missing_obs_or_var_filter
0.36s setup    tests/test_get_anndata.py::test_get_anndata_coords
0.35s setup    tests/test_get_anndata.py::test_get_anndata_value_filter
0.34s call     tests/test_open.py::test_open_soma_latest
0.32s call     tests/test_directory.py::test_get_census_version_description_errors
0.32s call     tests/test_open.py::test_can_open_with_anonymous_access
0.25s call     tests/test_open.py::test_opening_census_without_anon_access_fails_with_bogus_creds
0.03s setup    tests/test_directory.py::test_get_census_version_directory
0.03s call     tests/test_directory.py::test_get_census_version_directory
0.02s teardown tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-homo_sapiens]
0.02s teardown tests/test_acceptance.py::test_get_anndata[None-None-ctx_config10-homo_sapiens]
0.01s teardown tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-homo_sapiens]
0.01s setup    tests/test_acceptance.py::test_get_anndata[is_primary_data==True-None-ctx_config9-mus_musculus]
0.01s setup    tests/test_acceptance.py::test_get_anndata[None-None-ctx_config10-mus_musculus]
0.01s setup    tests/test_acceptance.py::test_get_anndata[cell_type=='neuron'-None-ctx_config8-mus_musculus]

(131 durations < 0.005s hidden.  Use -vv to show these durations.)
=============== 69 passed, 111 deselected in 15040.59s (4:10:40) ===============
```

### 2023-05-16

- Host: EC2 instance type: `r6id.32xlarge`, all nvme mounted as swap.
- Uname: Linux 5.19.0-1022-aws #23~22.04.1-Ubuntu SMP Fri Mar 17 15:38:24 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Python & census versions:

```python
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

```text
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

### 2023-03-29

**Config:**

- Host: EC2 instance type: `r6id.32xlarge`, all nvme mounted as swap.
- Uname: Linux bruce.aegea 5.15.0-1033-aws #37~20.04.1-Ubuntu SMP Fri Mar 17 11:39:30 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- Python & census versions:

```python
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

```text
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
