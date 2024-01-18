# Memento Pre-computed Estimator Cube Builder

The `cell_census_summary_cube` script pre-computes the estimators that are used by Memento, using the CELLxGENE Census
single-cell data. The estimators are output to a TileDB array named `estimators_cube`.

Usage instructions:

1. It is recommended to run this script on an AWS EC2 `r6id.24xlarge` instance running `Ubuntu 22.04`, 1024GB root drive, in the `us-west-2` region.
2. While the builder has been tuned to run within the available memory of this instance type, it is safest to configure the instance with swap space to avoid OOM errors. Copy this [script](https://github.com/chanzuckerberg/cellxgene-census/blob/d9bd1eb4a3e14974a0e7d9c23fb8368e79b92c2d/tools/scripts/aws/swapon_instance_storage.sh) to the instance and run as root: `sudo swapon_instance_storage.sh 1`. Note the `1` will only utilize one SSD for swap space, which should be sufficient.
3. Install Python: `sudo apt install python3-venv`
4. `git clone git@github.com:chanzuckerberg/cellxgene-census.git`
5. Setup a virtualenv and `pip install -r tools/models/memento/requirements.txt`.
6. Download the Census to local filesystem: `sudo aws s3 --no-sign-request sync s3://cellxgene-data-public/cell-census/<CENSUS_VERSION>/soma/ <LOCAL_PATH_TO_CENSUS_SOMA>`
6. To run: `python -O -m estimators_cube_builder --cube-uri <LOCAL_PATH_TO_CUBE>/ --experiment-uri <LOCAL_PATH_TO_CENSUS_SOMA>/census_data/homo_sapiens --overwrite --validate --consolidate 2>&1 | tee build-cube.log`.

For further performance, the local Census path can be on a volume mounted on SSD drive. E.g.:

```sh
sudo mkfs.ext4 -L census /dev/nvme1n2
sudo mkdir -p /mnt/census
sudo mount /dev/nvme1n2 /mnt/census
```

To inspect the results of the cube, see `estimators_cube_builder/cube-adhoc-query.ipynb`.

Notes:
* The scripts makes use of Python's multiprocessing to parallelize the estimator computations. The amount of memory used per sub-process and overall on the instance will be impacted by the constants `MIN_BATCH_SIZE`, `MAX_CELLS`, and `MAX_WORKERS`. The `MAX_CELLS` is the upper limit of cells that worker processes will be allowed to process at a given time (enforced by `ResourcePoolProcessExecutor`). This effectively controls the peak memory usage to avoid using swap space, which would negatively impact performance. However, if this causes the worker process count to be less than the CPU count, the CPUs will be underutilized. This can be rectified by decreasing the MIN_BATCH_SIZE, which will reduce the memory used per process and allow more workers processes to run in parallel.
* The script takes ~17 hours to run in the default configuration on the `r6id.24xlarge` instance size.
