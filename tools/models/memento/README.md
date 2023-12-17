# Memento Pre-computed Estimator Cube Builder

The `cell_census_summary_cube` script pre-computes the estimators that are used by Memento, using the CELLxGENE Census
single-cell data. The estimators are output to a TileDB array named `estimators_cube`.

Usage instructions:

1. It is recommended to run this script on an AWS EC2 `r6id.24xlarge` instance running `Ubuntu 22.04`, >=1024GB root drive, in the `us-west-2` region. The instance must be configured with swap space, making use of the available SSD drives. Copy this [script](https://github.com/chanzuckerberg/cellxgene-census/blob/d9bd1eb4a3e14974a0e7d9c23fb8368e79b92c2d/tools/scripts/aws/swapon_instance_storage.sh) to the instance and run as root: `sudo swapon_instance_storage.sh`.
2. Install Python: `sudo apt install python3-venv`
3. Setup a virtualenv and `pip install tiledbsoma psutil click`.
4. `git clone git@github.com:chanzuckerberg/cellxgene-census.git`
5. To run: `/usr/bin/time -v python -m tools/models/memento_builder s3://cellxgene-data-public/cell-census/2023-10-23/soma/census_data/homo_sapiens 2>&1 | tee ~/memento/cell_census_summary_cube.log`.
6. Consolidate the estimator cube:

```python
import tiledb
tiledb.consolidate('estimators_cube')
tiledb.vacuum('estimators_cube')
```

Optionally, but recommended for improved performance, replace step 5 with:
5.a. Download the Census to local filesystem: `sudo aws s3 --no-sign-request sync s3://cellxgene-data-public/cell-census/2023-10-23/soma/ <LOCAL_PATH_TO_CENSUS_SOMA>`
5.b. To run: `/usr/bin/time -v python -m tools/models/memento_builder <LOCAL_PATH_TO_CENSUS_SOMA>/census_data/homo_sapiens 2>&1 | tee ~/memento/cell_census_summary_cube.log`.

For further performance, the local Census path should be on a volume mounted on SSD drive. E.g.:

```sh
sudo mkfs.ext4 -L census /dev/nvme1n1
sudo mkdir -p /mnt/census
sudo mount /dev/nvme1n1 /mnt/census
```

To use or inspect the results as Pandas DataFrame:

```python
import tiledb
estimators = tiledb.open('estimators_cube').df[:]
```

Notes:

* The "size factors" are first computed for all cells (per cell) and stored in a TileDB Array called `obs_with_size_factor`. If the script is re-run, the size factors will be reloaded from this stored result. If you delete the `obs_with_size_factor` directory it will be recreated on the next run.
* The scripts makes use of Python's multiprocessing to parallelize the estimator computations. The amount of memory used per sub-process and overall on the instance will be impacted by the constants `MIN_BATCH_SIZE`, `MAX_CELLS`, and `MAX_WORKERS`. The `MAX_CELLS` is the number of cells that all worker processes combined can operate on at a time. This controls maximum memory usage to avoid swap usage, which will negatively impact performance. However, if this causes the worker process count to be less than the CPU count, the CPUs will be underutilized. This can be rectified by decreasing the MIN_BATCH_SIZE, which will reduce the memory used per process and allow more workers processes to run in parallel.
* The script takes ~30 hours to run in the default configuration on the `r6id.24xlarge` instance size.
