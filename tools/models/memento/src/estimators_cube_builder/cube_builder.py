from __future__ import annotations

import gc
import logging
import multiprocessing
import os
import shutil
from concurrent import futures
from typing import List, Tuple, cast

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import scipy.sparse
import tiledb
import tiledbsoma as soma
from somacore import AxisQuery, ExperimentAxisQuery
from tiledbsoma import SOMATileDBContext

from .cube_schema import (
    CUBE_DIMS_VAR,
    CUBE_LOGICAL_DIMS_OBS,
    CUBE_TILEDB_ATTRS_OBS,
    ESTIMATOR_NAMES,
    build_cube_schema,
)
from .cube_validator import validate_cube
from .estimators import bin_size_factor, compute_mean, compute_sem, compute_sev, compute_variance, gen_multinomial
from .mp import create_resource_pool_executor

PROFILE_MODE = bool(os.getenv("PROFILE_MODE", False))  # Run pass 2 in single-process mode with profiling output

# TODO: parameterize constants below

OBS_SIZE_FACTORS_ARRAY_SUFFIX = "_size_factors"

TILEDB_SOMA_BUFFER_BYTES = 2**31

# The minimum number of cells that should be processed at a time by each child process.
MIN_BATCH_SIZE = 2**13

Q = 0.1  # RNA capture efficiency depending on technology

MAX_WORKERS = None  # None means use multiprocessing's dynamic default

# The maximum number of cells values to be processed at any given time ("X nnz per batch" would be a better metric
# due to differences in X sparsity across cells, but it is not efficient to compute). The multiprocessing logic will
# not submit new jobs while this value is exceeded, thereby keeping memory usage bounded. This is needed since job
# sizes vary considerably in their memory usage, due to the high cell count of some batches (if batch sizes were not
# highly variable, we could just limit by process/worker count).
MAX_CELLS = 512_000

OBS_VALUE_FILTER = "is_primary_data == True"


logging.basicConfig(
    format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.captureWarnings(True)


# pd.options.display.max_columns = None
# pd.options.display.width = 1024
# pd.options.display.min_rows = 40


def compute_all_estimators_for_obs_group(obs_group_rows: pd.DataFrame, obs_df: pd.DataFrame) -> pd.Series[float]:
    """Computes all estimators for a given obs group's expression values"""
    obs_group_name = cast(Tuple[str, ...], obs_group_rows.name)

    # Filter obs to the rows of the current group, and retrieve the "size factors" data for those rows. The full set
    # of rows in the group is needed to create "dense" arrays used to compute estimators each gene in
    # compute_all_estimators_for_gene(), where the full set of obs rows can no longer be determined from the "sparse"
    # set of expression values for a given gene.
    size_factors_for_obs_group = obs_df[["approx_size_factor"]].loc[obs_group_rows[[]].index.drop_duplicates()]

    gene_groups = obs_group_rows.groupby(CUBE_DIMS_VAR, observed=True)
    estimators = gene_groups.apply(
        lambda gene_group_rows: compute_all_estimators_for_gene(
            obs_group_name, gene_group_rows, size_factors_for_obs_group
        )
    )

    assert (
        estimators.index.nunique() == obs_group_rows["feature_id"].nunique()
    ), f"estimators count incorrect in group {obs_group_name}"
    return estimators  # type: ignore


def compute_all_estimators_for_gene(
    gene_group_name: Tuple[str, ...], gene_group_rows: pd.DataFrame, size_factors_for_obs_group: pd.DataFrame
) -> pd.Series[float]:
    """Computes all estimators for a given {<dim1>, ..., <dimN>, gene} group of expression values"""

    data_dense, X_dense, size_factors_dense = dense_gene_data(gene_group_rows, size_factors_for_obs_group)

    X_csc, X_sparse = sparse_gene_data(data_dense)

    n_obs = len(X_dense)
    if n_obs == 0:
        return pd.Series(data=[0.0] * len(ESTIMATOR_NAMES), dtype=float)

    nnz = gene_group_rows.shape[0]
    min_ = X_sparse.min()
    max_ = X_sparse.max()
    sum_ = X_sparse.sum()
    mean = compute_mean(X_dense, size_factors_dense)
    sem = compute_sem(X_dense, size_factors_dense)
    variance = compute_variance(X_csc, Q, size_factors_dense, group_name=gene_group_name)
    sev, selv = compute_sev(X_csc, Q, size_factors_dense, num_boot=500, group_name=gene_group_name)

    estimators = dict(
        n_obs=n_obs, nnz=nnz, min=min_, max=max_, sum=sum_, mean=mean, sem=sem, var=variance, sev=sev, selv=selv
    )

    return pd.Series(data=[estimators[n] for n in ESTIMATOR_NAMES], dtype=np.float64)


def sparse_gene_data(data_dense: pd.DataFrame) -> Tuple[scipy.sparse.csc_matrix, npt.NDArray[np.float32]]:
    data_sparse = data_dense[data_dense.soma_data.notna()]
    X_sparse = data_sparse.soma_data.to_numpy()
    X_csc = scipy.sparse.coo_array(
        (X_sparse, (data_sparse.index, np.zeros(len(data_sparse), dtype=int))), shape=(len(data_dense), 1)
    ).tocsc()
    return X_csc, X_sparse


def dense_gene_data(
    gene_group_rows: pd.DataFrame, size_factors_for_obs_group: pd.DataFrame
) -> Tuple[pd.DataFrame, npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    data_dense = pd.concat(
        [size_factors_for_obs_group[[]], gene_group_rows[["soma_data"]]], axis=1, copy=False
    ).reset_index()

    X_dense = data_dense.soma_data.to_numpy()
    X_dense = np.nan_to_num(X_dense)
    size_factors_dense = size_factors_for_obs_group.approx_size_factor.to_numpy()

    return data_dense, X_dense, size_factors_dense


def compute_all_estimators_for_batch_tdb(
    soma_dim_0: List[int], obs_df: pd.DataFrame, var_df: pd.DataFrame, X_uri: str, batch: int, cube_uri: str
) -> pd.DataFrame:
    """Compute estimators for each gene"""

    with soma.SparseNDArray.open(
        X_uri,
        context=soma.SOMATileDBContext().replace(
            tiledb_config={
                "soma.init_buffer_bytes": TILEDB_SOMA_BUFFER_BYTES,
                "vfs.s3.region": "us-west-2",
                "vfs.s3.no_sign_request": True,
            }
        ),
    ) as X:
        X_df = X.read(coords=(soma_dim_0, var_df.index.values)).tables().concat().to_pandas()
        logging.info(f"Pass 2: Start X batch {batch}, cells={len(soma_dim_0)}, nnz={len(X_df)}")
        result = compute_all_estimators_for_batch_pd(X_df, obs_df, var_df)
        if len(result) == 0:
            logging.warning(f"Pass 2: Batch {batch} had empty result, cells={len(soma_dim_0)}, nnz={len(X_df)}")
        logging.info(f"Pass 2: End X batch {batch}, cells={len(soma_dim_0)}, nnz={len(X_df)}")

        assert all(result.index.value_counts() <= 1), "tiledb batch has repeated cube rows"

    write_estimators_batch(result, obs_df, cube_uri)

    gc.collect()

    return result


def compute_all_estimators_for_batch_pd(X_df: pd.DataFrame, obs_df: pd.DataFrame, var_df: pd.DataFrame) -> pd.DataFrame:
    result = (
        X_df.set_index("soma_dim_1")
        .join(var_df[CUBE_DIMS_VAR])
        .set_index("soma_dim_0")
        .join(obs_df[CUBE_LOGICAL_DIMS_OBS])
        .groupby(CUBE_LOGICAL_DIMS_OBS, observed=True, sort=False)
        .apply(lambda obs_group: compute_all_estimators_for_obs_group(obs_group, obs_df))
        .rename(mapper=dict(enumerate(ESTIMATOR_NAMES)), axis=1)
    )
    return result


# TODO: replace this with obs.raw_sum
def sum_gene_expression_levels_by_cell(X_tbl: pa.Table, batch: int) -> pd.Series[float]:
    logging.info(f"Pass 1: Computing X batch {batch}, nnz={X_tbl.shape[0]}")

    # TODO: use PyArrow API only; avoid Pandas conversion
    result = X_tbl.to_pandas()[["soma_dim_0", "soma_data"]].groupby("soma_dim_0", sort=False).sum()["soma_data"]

    logging.info(f"Pass 1: Computing X batch {batch}, nnz={X_tbl.shape[0]}: done")

    return result  # type: ignore


def pass_1_compute_size_factors(query: ExperimentAxisQuery, layer: str) -> pd.DataFrame:
    obs_df = (
        query.obs(column_names=["soma_joinid"] + CUBE_LOGICAL_DIMS_OBS).concat().to_pandas().set_index("soma_joinid")
    )
    obs_df["size_factor"] = 0  # accumulated

    executor = futures.ThreadPoolExecutor()
    summing_futures = []
    X_rows = query._ms.X[layer].shape[0]
    cum_rows = 0
    for n, X_tbl in enumerate(query.X(layer).tables(), start=1):
        cum_rows += X_tbl.shape[0]
        logging.info(f"Pass 1: Submitting X batch {n}, nnz={X_tbl.shape[0]}, {100 * cum_rows / X_rows:0.1f}%")
        summing_futures.append(executor.submit(sum_gene_expression_levels_by_cell, X_tbl, n))

    for n, summing_future in enumerate(futures.as_completed(summing_futures), start=1):
        # Accumulate cell sums, since a given cell's X values may be returned across multiple tables
        cell_sums = summing_future.result()
        obs_df["size_factor"] = obs_df["size_factor"].add(cell_sums, fill_value=0)
        logging.info(f"Pass 1: Completed {n} of {len(summing_futures)} batches, " f"total cube rows={len(obs_df)}")

    # Convert size factors to relative - prevents small floats for variance
    global_n_umi = obs_df["size_factor"].values.mean()
    obs_df["size_factor"] = obs_df["size_factor"].values / global_n_umi

    # Bin all sums to have fewer unique values, to speed up bootstrap computation
    obs_df["approx_size_factor"] = bin_size_factor(obs_df["size_factor"].values)

    return cast(pd.DataFrame, obs_df[CUBE_LOGICAL_DIMS_OBS + ["approx_size_factor"]])


def pass_2_compute_estimators(
    cube_uri: str, query: ExperimentAxisQuery, size_factors: pd.DataFrame, /, measurement_name: str, layer: str
) -> None:
    var_df = query.var().concat().to_pandas().set_index("soma_joinid")
    obs_df = (
        query.obs(column_names=["soma_joinid"] + CUBE_LOGICAL_DIMS_OBS).concat().to_pandas().set_index("soma_joinid")
    )

    obs_df = obs_df.join(size_factors[["approx_size_factor"]])

    if tiledb.array_exists(cube_uri):
        logging.info("Pass 2: Resuming from existing estimators cube")
        with tiledb.open(cube_uri, mode="r") as estimators_cube:
            df = estimators_cube.query(attrs=CUBE_TILEDB_ATTRS_OBS).df[:][CUBE_LOGICAL_DIMS_OBS]
            existing_groups = df.drop_duplicates()
            existing_groups = existing_groups.set_index(list(existing_groups.columns))
    else:
        existing_groups = None
        tiledb.Array.create(cube_uri, build_cube_schema(obs_df))
        logging.info("Pass 2: Created new estimators cube")

    # Process X by cube rows. This ensures that estimators are computed
    # for all X data contributing to a given cube row aggregation.
    # Note: `groups` converts categoricals to strs, which is not memory-efficient
    cube_obs_coords = obs_df[CUBE_LOGICAL_DIMS_OBS].groupby(CUBE_LOGICAL_DIMS_OBS, observed=True)
    cube_obs_coord_groups = cube_obs_coords.groups

    soma_dim_0_batch: List[int] = []
    batch_futures = []
    n = n_cum_cells = 0

    executor = create_resource_pool_executor(max_workers=MAX_WORKERS, max_resources=MAX_CELLS)

    n_total_cells = query.n_obs

    # For testing/debugging: Run pass 2 without multiprocessing
    if PROFILE_MODE:
        # force numba jit compilation outside of profiling
        gen_multinomial(np.array([1, 1, 1]), 3, 1)

        import cProfile

        def process_batch() -> None:
            nonlocal n
            n += 1
            batch_result = compute_all_estimators_for_batch_tdb(
                soma_dim_0_batch, obs_df, var_df, query.experiment.ms[measurement_name].X[layer].uri, n, cube_uri
            )
            if len(batch_result) > 0:
                batch_result = batch_result.reset_index(CUBE_LOGICAL_DIMS_OBS)

                # NOTE: The Pandas categorical columns must have the same underlying dictionaries as the TileDB Array
                #  schema's enumeration columns
                for col_ in CUBE_LOGICAL_DIMS_OBS:
                    # TODO: DRY up category values generation with build_cube_schema (they must be equivalent)
                    batch_result[col_] = pd.Categorical(
                        batch_result[col_], categories=obs_df[col_].unique().astype(str)
                    )

                tiledb.from_pandas(cube_uri, batch_result, mode="append")

        with cProfile.Profile() as pr:
            for soma_dim_0_ids in cube_obs_coord_groups.values():
                soma_dim_0_batch.extend(soma_dim_0_ids)
                if len(soma_dim_0_batch) < MIN_BATCH_SIZE:
                    continue

                process_batch()
                soma_dim_0_batch = []

            if len(soma_dim_0_batch) > 0:
                process_batch()

            pr.dump_stats(f"/tmp/pass_2_compute_estimators_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.prof")

    else:  # use multiprocessing

        def submit_batch(soma_dim_0_batch_: List[int]) -> None:
            nonlocal n, n_cum_cells
            n += 1
            n_cum_cells += len(soma_dim_0_batch_)

            X_uri = query.experiment.ms[measurement_name].X[layer].uri

            logging.info(
                f"Pass 2: Submitting cells batch {n}, cells={len(soma_dim_0_batch_)}, "
                f"{100 * n_cum_cells / n_total_cells:0.1f}%"
            )

            batch_futures.append(
                executor.submit(
                    len(soma_dim_0_batch_),
                    compute_all_estimators_for_batch_tdb,
                    soma_dim_0_batch_,
                    obs_df,
                    var_df,
                    X_uri,
                    n,
                    cube_uri,
                )
            )

        for group_key, soma_dim_0_ids in cube_obs_coord_groups.items():
            if existing_groups is None or group_key not in existing_groups.index:
                soma_dim_0_batch.extend(soma_dim_0_ids)
            else:
                logging.info(f"Pass 2: Group {group_key} already computed. Skipping computation.")
                continue

            # Fetch data for multiple cube rows at once, to reduce X.read() call count
            if len(soma_dim_0_batch) < MIN_BATCH_SIZE:
                continue

            submit_batch(soma_dim_0_batch)
            soma_dim_0_batch = []

        # Process final batch
        if len(soma_dim_0_batch) > 0:
            submit_batch(soma_dim_0_batch)

        # Accumulate results

        n_cum_cells = 0
        for n, future in enumerate(futures.as_completed(batch_futures), start=1):
            future.result()
            logging.info(
                f"Pass 2: Completed {n} of {len(batch_futures)} batches ({100 * n / len(batch_futures):0.1f}%)"
            )
            gc.collect()

        logging.info("Pass 2: Completed")


def write_estimators_batch(batch_result: pd.DataFrame, obs_df: pd.DataFrame, cube_uri: str) -> None:
    if len(batch_result) > 0:
        batch_result = batch_result.reset_index(CUBE_LOGICAL_DIMS_OBS)

        # TODO: Can remove once https://github.com/TileDB-Inc/TileDB-Py/issues/1879 fix is available
        # Ensure Pandas categorical columns must have the same underlying dictionaries as the TileDB Array
        #  schema's enumeration columns
        for col_ in CUBE_LOGICAL_DIMS_OBS:
            batch_result[col_] = pd.Categorical(batch_result[col_], categories=obs_df[col_].unique().astype(str))

        logging.info("Pass 2: Writing to estimator cube.")
        tiledb.from_pandas(cube_uri, batch_result, mode="append")

    else:
        logging.warning("Pass 2: Batch had empty result")


def build(
    cube_uri: str, experiment_uri: str, measurement_name: str = "RNA", layer: str = "raw", validate: bool = True
) -> bool:
    # init multiprocessing
    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)

    soma_ctx = SOMATileDBContext(
        tiledb_config={
            "vfs.s3.region": os.getenv("AWS_REGION", "us-west-2"),
            "vfs.s3.no_sign_request": True,
        }
    )
    with soma.Experiment.open(uri=experiment_uri, context=soma_ctx) as exp:
        query = exp.axis_query(
            measurement_name=measurement_name,
            obs_query=AxisQuery(value_filter=OBS_VALUE_FILTER),
        )
        logging.info(f"Pass 1: Processing {query.n_obs} cells and {query.n_vars} genes")

        obs_size_factors_uri = cube_uri + OBS_SIZE_FACTORS_ARRAY_SUFFIX
        if not tiledb.array_exists(obs_size_factors_uri):
            logging.info("Pass 1: Compute Approx Size Factors")
            size_factors = pass_1_compute_size_factors(query, layer)

            size_factors = size_factors.astype({col: "category" for col in CUBE_LOGICAL_DIMS_OBS})
            tiledb.from_pandas(obs_size_factors_uri, size_factors.reset_index(), index_col=[0])
            logging.info("Saved `obs_with_size_factor` TileDB Array")
        else:
            logging.info("Pass 1: Compute Approx Size Factors (loading from stored data)")
            size_factors = tiledb.open(obs_size_factors_uri).df[:].set_index("soma_joinid")

        logging.info("Pass 2: Compute Estimators")
        query = exp.axis_query(
            measurement_name=measurement_name,
            obs_query=AxisQuery(value_filter=OBS_VALUE_FILTER),
        )
        logging.info(f"Pass 2: Processing {query.n_obs} cells and {query.n_vars} genes")

        pass_2_compute_estimators(cube_uri, query, size_factors, measurement_name=measurement_name, layer=layer)

    if validate:
        logging.info("Validating estimators cube")
        validate_cube(cube_uri, experiment_uri)  # raises exception if invalid
        logging.info("Validation complete")

    return True


@click.command()
@click.option("--cube-uri")
@click.option("--experiment-uri")
@click.option("--measurement_name", default="RNA")
@click.option("--layer", default="raw")
@click.option("--validate/--no-validate", is_flag=True, default=True)
@click.option("--resume", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
def build_cli(
    cube_uri: str, experiment_uri: str, measurement_name: str, layer: str, validate: bool, resume: bool, overwrite: bool
) -> None:
    if resume and overwrite:
        raise ValueError("Cannot specify both --resume and --overwrite")

    if tiledb.array_exists(cube_uri):
        if resume:
            logging.info(f"Resuming from existing estimators cube at {cube_uri}.")
        elif overwrite:
            logging.info(f"Overwriting existing estimators cube at {cube_uri}.")
            shutil.rmtree(cube_uri)
        else:
            logging.error(
                "Estimators cube already exists and neither --resume or --overwrite options specified. Exiting."
            )
            exit(1)

    build(cube_uri, experiment_uri, measurement_name, layer, validate)


if __name__ == "__main__":
    build_cli()
