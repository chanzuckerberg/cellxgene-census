#!/usr/bin/env python
import cProfile
import glob
import itertools
import json
import logging
import os
import pstats
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, process
from functools import partial, wraps, reduce
from typing import List, Tuple, cast, Optional, Dict, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats
import tiledb
#from sklearn.linear_model import LinearRegression
from sklearnex.linear_model import LinearRegression
import polars as pl

OBS_GROUPS_ARRAY = "obs_groups"
ESTIMATORS_ARRAY = "estimators"

CUBE_LOGICAL_DIMS_OBS = [
    "cell_type_ontology_term_id",
    "dataset_id",
    "tissue_general_ontology_term_id",
    "assay_ontology_term_id",
    "donor_id",
    "disease_ontology_term_id",
    "sex_ontology_term_id",
    "development_stage_ontology_term_id",
    "self_reported_ethnicity_ontology_term_id",
    "suspension_type",
]

fn_cum_time: Dict[str, float] = defaultdict(lambda: 0)
fn_calls: Dict[str, int] = defaultdict(lambda: 0)


def timeit_report(func):
    @wraps(func)
    def timeit_report_wrapper(*args, **kwargs):
        # return func(*args, **kwargs), None

        # with cProfile.Profile() as prof:
        #     result = func(*args, **kwargs)

        #     f = tempfile.mkstemp()[1]
        #     prof.dump_stats(f)

        result = func(*args, **kwargs)

        f = None

        sorted_fn_names = [k for k, _ in sorted(fn_cum_time.items(), key=lambda i: i[1], reverse=True)]
        for fn_name in sorted_fn_names:
            print(f'[timing {os.getpid()}] {fn_name}: '
                  f'cum_time={fn_cum_time[fn_name]} sec; avg_time={(fn_cum_time[fn_name] / fn_calls[fn_name]):.3f}; '
                  f'calls={fn_calls[fn_name]}')

        return result, f

    return timeit_report_wrapper


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        exec_time = end_time - start_time

        fn_name = func.__name__
        fn_cum_time[fn_name] += exec_time
        fn_calls[fn_name] += 1
        # print(f'[timing] {fn_name}: exec time={exec_time:.3f} sec; '
        #       f'cum_time={fn_cum_time[fn_name]} sec; avg_time={(fn_cum_time[fn_name] / fn_calls[fn_name]):.3f}; '
        #       f'calls={fn_calls[fn_name]}')

        return result
    return timeit_wrapper


@timeit
def query_estimators(cube_path: str, obs_groups_df: pd.DataFrame, features: List[str]) -> pl.DataFrame:
    tiledb_config = {
        "soma.init_buffer_bytes": 2**31,
    }
    with tiledb.open(os.path.join(cube_path, ESTIMATORS_ARRAY), "r", config=tiledb_config) as estimators_array:
        estimators_df = pl.DataFrame(estimators_array.df[features, obs_groups_df.obs_group_joinid.values])

    # TODO: Determine whether it's reasonable to drop these values, or if we should revisit how they're being
    #  computed in the first place. If reasonable, this filtering should be done by the cube builder, not here.
    # This filtering ensures that we will not take of logs of non-positive values, or end up with selm values of 0
    estimators_df = drop_invalid_data(estimators_df)

    return cast(pl.DataFrame, estimators_df)


@timeit
def drop_invalid_data(estimators_df: pl.DataFrame) -> pl.DataFrame:
    drop_mask = (estimators_df["sem"] <= 0) | (estimators_df["sem"] >= estimators_df["mean"])
    if drop_mask.any():
        logging.warning(f"dropping {drop_mask.sum()} rows with invalid values ({drop_mask.sum() / len(drop_mask):.2%})")
        estimators_df = estimators_df.filter(~drop_mask)
    return estimators_df


def compute_all(cube_path: str, query_filter: str, treatment: str, n_processes: int, n_features: Optional[int] = None) -> Tuple[pd.DataFrame, pstats.Stats]:
    with tiledb.open(os.path.join(cube_path, OBS_GROUPS_ARRAY), "r") as obs_groups_array:
        obs_groups_df = obs_groups_array.query(cond=query_filter or None).df[:]

        distinct_treatment_values = obs_groups_df[treatment].nunique()
        assert distinct_treatment_values == 2, "treatment must have exactly 2 distinct values"

        # convert categorical columns to ints
        for col in obs_groups_df.select_dtypes(include=["category"]).columns:
            obs_groups_df[col] = obs_groups_df[col].cat.codes

    # TODO: need canonical list of features efficiently
    features = get_features(cube_path)
    if n_features is not None:
        rng = np.random.default_rng(1024)
        features = rng.choice(features, size=n_features, replace=False)

    # compute each feature group in parallel
    n_feature_groups = min(len(features), n_processes)
    feature_groups = [features.tolist() for features in np.array_split(np.array(features), n_feature_groups)]
    print(
        f"computing for {len(obs_groups_df)} obs groups ({obs_groups_df.n_obs.sum()} cells) and {n_features} features using {n_feature_groups} processes, {len(features) // n_feature_groups} features/process"
    )

    # make treatment variable be in the first column of the design matrix
    variables = [treatment] + [covariate for covariate in CUBE_LOGICAL_DIMS_OBS if covariate != treatment]
    design = pd.get_dummies(obs_groups_df[variables], drop_first=True, dtype=int)

    result_groups = ProcessPoolExecutor(max_workers=n_processes).map(
        partial(compute_for_features, cube_path, design, obs_groups_df), feature_groups, range(len(feature_groups))
    )

    results = list(result_groups)
    data = itertools.chain.from_iterable([r[0] for r in results])  # flatten results
    stats = reduce(lambda s1, s2: s1.add(s2), [pstats.Stats(r[1]) if r[1] else pstats.Stats() for r in results])

    return pd.DataFrame(data, columns=["feature_id", "coef", "z", "pval"], copy=False).set_index("feature_id"), stats


def get_features(cube_path: str) -> List[str]:
    feature_id_path = os.path.join(cube_path, "feature_ids.json")
    if os.path.isfile(feature_id_path):
        with open(feature_id_path) as f:
            features = json.load(f)
    else:
        with tiledb.open(
            os.path.join(cube_path, ESTIMATORS_ARRAY), "r", config={"soma.init_buffer_bytes": 2**32}
        ) as estimators_array:
            features = (
                estimators_array.query(attrs=[], dims=["feature_id"]).df[:]["feature_id"].drop_duplicates().tolist()
            )
            with open(feature_id_path, "w") as f:
                json.dump(features, f)
    return cast(List[str], features)


@timeit_report
def compute_for_features(
    cube_path: str, design: pd.DataFrame, obs_groups_df: pd.DataFrame, features: List[str], feature_group_key: int
) -> List[Tuple[str, np.float32, np.float32, np.float32]]:
    print(f"computing for feature group {feature_group_key}, n={len(features)}, {features[0]}..{features[-1]}...")
    estimators = query_estimators(cube_path, obs_groups_df, features)

    cell_counts = obs_groups_df["n_obs"].values
    obs_group_joinids = obs_groups_df[["obs_group_joinid"]]

    result = [
        (feature_id, *compute_for_feature(cell_counts, design, feature_estimators, obs_group_joinids))  # type:ignore
        for feature_id, feature_estimators in estimators.group_by(['feature_id'])
    ]

    print(f"computed for feature group {feature_group_key}, {features[0]}..{features[-1]}")

    return result


@timeit
def compute_for_feature(
    cell_counts: npt.NDArray[np.float32],
    design: pd.DataFrame,
    estimators: pd.DataFrame,
    obs_group_joinids: pd.DataFrame
) -> Tuple[np.float32, np.float32, np.float32]:
    # ensure estimators are available for all obs groups (for when feature had no expression data for some obs groups)
    estimators = fill_missing_data(obs_group_joinids, estimators)

    assert len(estimators) == len(design)

    # Transform to log space (alternatively can resample in log space)
    lm, selm = transform_to_log_space(estimators["mean"].to_numpy(), estimators["sem"].to_numpy())

    return de_wls(X=design.values, y=lm, n=cell_counts, v=selm**2)


@timeit
def transform_to_log_space(
    m: npt.NDArray[np.float32], sem: npt.NDArray[np.float32]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    lm = np.log(m)
    selm = (np.log(m + sem) - np.log(m - sem)) / 2
    assert (selm > 0).all()
    return lm, selm


@timeit
def fill_missing_data(obs_group_joinids: pd.DataFrame, feature_estimators: pl.DataFrame) -> pl.DataFrame:
    feature_estimators = pl.DataFrame(obs_group_joinids).join(feature_estimators[["obs_group_joinid", "mean", "sem"]], on="obs_group_joinid", how="left")

    return feature_estimators.with_columns(feature_estimators["mean"].fill_null(1e-3), feature_estimators["sem"].fill_null(1e-4))


@timeit
def de_wls_fit(X: npt.NDArray[np.float32], y: npt.NDArray[np.float32], n: npt.NDArray[np.float32]) -> np.float32:
    # fit WLS using sample_weights
    WLS = LinearRegression()
    WLS.fit(X, y, sample_weight=n)

    # note: we have all the other coefficients (i.e. effect size) for the other covariates here as well, but we only
    # want the treatment effect for now
    return cast(np.float32, WLS.coef_[0])


@timeit
def de_wls_stats(
    X: npt.NDArray[np.float32], v: npt.NDArray[np.float32], coef: np.float32
) -> Tuple[np.float32, np.float32]:
    W = de_wls_stats_W(v)
    m = de_wls_stats_matmul(W, X)
    pinv = de_wls_stats_pinv(m)
    beta_var_hat = np.diag(pinv)
    se = np.sqrt(beta_var_hat[0])

    z = coef / se
    pv = stats.norm.sf(np.abs(z)) * 2

    return z, pv


@timeit
def de_wls_stats_pinv(m):
    return np.linalg.pinv(m)


@timeit
def de_wls_stats_matmul(W, X):
    m = (X.T * W) @ X
    return m


@timeit
def de_wls_stats_W(v):
    return (1/v) #.reshape((-1, 1))


@timeit
def de_wls(
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    n: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
) -> Tuple[np.float32, np.float32, np.float32]:
    """
    Perform DE for each gene using Weighted Least Squares (i.e., a weighted Linear Regression model)
    """
    coef = de_wls_fit(X, y, n)
    z, pv = de_wls_stats(X, v, coef)

    return coef, z, pv


# Script entrypoint
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python diff_expr.py <filter> <treatment> <cube_path> <csv_output_path> <n_processes> <n_features>")
        sys.exit(1)

    filter_arg, treatment_arg, cube_path_arg, n_processes, n_features = sys.argv[1:6]

    de_result = compute_all(cube_path_arg, filter_arg, treatment_arg, int(n_processes), int(n_features))

    # Output DE result
#    print(de_result)
#    de_result.to_csv(csv_output_path_arg)

