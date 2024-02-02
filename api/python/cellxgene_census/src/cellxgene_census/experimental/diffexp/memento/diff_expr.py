#!/usr/bin/env python
import itertools
import json
import logging
import os
import pstats
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial, reduce
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import scipy.stats as stats
import tiledb
from sklearn.linear_model import LinearRegression

OBS_GROUPS_ARRAY = "obs_groups"
ESTIMATORS_ARRAY = "estimators"
FEATURE_IDS_FILE = "feature_ids.json"

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


def compute_memento_estimators_from_precomputed_stats(estimators_df: pl.DataFrame) -> pl.DataFrame:
    # mean: (X.sum() + 1) / (size_factors.sum() + 1))
    # sem: (X.std() * np.sqrt(n_obs)) / size_factors.sum())
    n_obs = estimators_df["n_obs"].to_numpy()
    expr_sum = estimators_df["sum"].to_numpy()
    expr_sumsq = estimators_df["sumsq"].to_numpy()
    size_factors = estimators_df["size_factor"].to_numpy()
    mean = (expr_sum + 1) / (size_factors + 1)
    var = expr_sumsq / n_obs - (expr_sum / n_obs) ** 2
    var[var < 0] = 0  # ensure variances are non-negative
    sem = np.sqrt(var) * np.sqrt(n_obs) / size_factors

    estimators_df = estimators_df.with_columns([pl.Series("mean", mean), pl.Series("sem", sem)])
    return estimators_df


# @timeit
def query_estimators(
    cube_path: str,
    obs_groups_df: pd.DataFrame,
    features: List[str],
) -> pl.DataFrame:
    tiledb_config = {
        "py.init_buffer_bytes": 2**31,
    }
    with tiledb.open(os.path.join(cube_path, ESTIMATORS_ARRAY), "r", config=tiledb_config) as estimators_array:
        estimators_df = estimators_array.df[features, obs_groups_df.obs_group_joinid.values]
        estimators_df = (
            estimators_df.merge(
                obs_groups_df[["obs_group_joinid", "selected_vars_group_joinid"]], on="obs_group_joinid"
            )
            .groupby(["feature_id", "selected_vars_group_joinid"])
            .sum()
            .reset_index()
        )
        estimators_df["obs_group_joinid"] = estimators_df["selected_vars_group_joinid"].astype("uint32")
        del estimators_df["selected_vars_group_joinid"]
        print(estimators_df.dtypes)
        estimators_df = pl.DataFrame(estimators_df)

    estimators_df = compute_memento_estimators_from_precomputed_stats(estimators_df)
    # TODO: Determine whether it's reasonable to drop these values, or if we should revisit how they're being
    #  computed in the first place. If reasonable, this filtering should be done by the cube builder, not here.
    # This filtering ensures that we will not take of logs of non-positive values, or end up with selm values of 0
    estimators_df = drop_invalid_data(estimators_df)

    return cast(pl.DataFrame, estimators_df)


# @timeit
def drop_invalid_data(estimators_df: pl.DataFrame) -> pl.DataFrame:
    drop_mask = (estimators_df["sem"] <= 0) | (estimators_df["sem"] >= estimators_df["mean"])
    if drop_mask.any():
        logging.warning(f"dropping {drop_mask.sum()} rows with invalid values ({drop_mask.sum() / len(drop_mask):.2%})")
        estimators_df = estimators_df.filter(~drop_mask)
    return estimators_df


def compute_all(
    cube_path: str,
    query_filter: str,
    treatment: str,
    n_processes: int,
    covariates: Optional[List[str]] = ["dataset_id"],
) -> Tuple[pd.DataFrame, pstats.Stats]:
    with tiledb.open(os.path.join(cube_path, OBS_GROUPS_ARRAY), "r") as obs_groups_array:
        obs_groups_df = obs_groups_array.query(cond=query_filter or None).df[:]
        if covariates:
            obs_groups_df = obs_groups_df[covariates + [treatment, "obs_group_joinid", "n_obs"]]
        else:
            covariates = CUBE_LOGICAL_DIMS_OBS

        distinct_treatment_values = obs_groups_df[treatment].nunique()
        assert distinct_treatment_values == 2, "treatment must have exactly 2 distinct values"

    features = get_features(cube_path, None)

    # compute each feature group in parallel
    n_feature_groups = min(len(features), n_processes)
    feature_groups = [features.tolist() for features in np.array_split(np.array(features), n_feature_groups)]
    logging.debug(
        f"computing for {len(obs_groups_df)} obs groups ({obs_groups_df.n_obs.sum()} cells) and {len(features)} features using {n_feature_groups} processes, {len(features) // n_feature_groups} features/process"
    )

    # make treatment variable be in the first column of the design matrix
    variables = [treatment] + [covariate for covariate in covariates if covariate != treatment]

    agg_dict = {i: "first" for i in variables}
    agg_dict["n_obs"] = "sum"
    selected_vars_groups_groupby = obs_groups_df.groupby(variables, observed=True)
    selected_vars_groups_df = selected_vars_groups_groupby.agg(agg_dict)
    obs_groups_df["selected_vars_group_joinid"] = selected_vars_groups_groupby.ngroup().astype("uint32")
    selected_vars_groups_df["obs_group_joinid"] = np.arange(len(selected_vars_groups_df), dtype="uint32")

    design = pd.get_dummies(selected_vars_groups_df[variables].astype(str), drop_first=True, dtype=int)
    assert design.shape[1] == selected_vars_groups_df[variables].nunique().sum() - len(variables)

    result_groups = ProcessPoolExecutor(max_workers=n_processes).map(
        partial(
            compute_for_features,
            cube_path,
            design,
            obs_groups_df[["obs_group_joinid", "selected_vars_group_joinid", "n_obs"]],
            selected_vars_groups_df[["obs_group_joinid", "n_obs"]],
        ),
        feature_groups,
        range(len(feature_groups)),
    )

    results = list(result_groups)
    assert len(results)

    # HACK: handle tuple-typed rests when @cprofile decorator is used on compute_for_features()
    if isinstance(results[0], tuple):  # type:ignore
        # flatten results
        data = itertools.chain.from_iterable([r[0] for r in results])  # type: ignore[unreachable]
        stats = reduce(lambda s1, s2: s1.add(s2), [pstats.Stats(r[1]) if r[1] else pstats.Stats() for r in results])
    else:
        data = itertools.chain.from_iterable(results)  # flatten results
        stats = pstats.Stats()

    results = pd.DataFrame(data, columns=["feature_id", "coef", "z", "pval"], copy=False).set_index("feature_id")
    results.sort_values("coef", ascending=False, inplace=True)
    return results, stats


def get_features(cube_path: str, n_features: Optional[int] = None) -> List[str]:
    with open(os.path.join(cube_path, FEATURE_IDS_FILE)) as f:
        feature_ids = json.load(f)

    if n_features is not None:
        # for testing purposes, useful to limit the number of features
        rng = np.random.default_rng(1024)
        feature_ids = rng.choice(feature_ids, size=n_features, replace=False)

    return cast(List[str], feature_ids)


# @cprofile
# @timeit_report
def compute_for_features(
    cube_path: str,
    design: pd.DataFrame,
    obs_groups_df: pd.DataFrame,
    selected_vars_groups_df: pd.DataFrame,
    features: List[str],
    feature_group_key: int,
) -> List[Tuple[str, np.float32, np.float32, np.float32]]:
    logging.debug(
        f"computing for feature group {feature_group_key}, n={len(features)}, {features[0]}..{features[-1]}..."
    )
    estimators = query_estimators(cube_path, obs_groups_df, features)
    estimators = estimators.with_columns(estimators["obs_group_joinid"].cast(pl.UInt32))
    cell_counts = selected_vars_groups_df["n_obs"].values
    obs_group_joinids = selected_vars_groups_df[["obs_group_joinid"]]

    result = [
        (feature_id, *compute_for_feature(cell_counts, design, feature_estimators, obs_group_joinids))  # type:ignore
        for feature_id, feature_estimators in estimators.group_by(["feature_id"])
    ]

    logging.debug(f"computed for feature group {feature_group_key}, {features[0]}..{features[-1]}")

    return result


# @timeit
def compute_for_feature(
    cell_counts: npt.NDArray[np.float32],
    design: pd.DataFrame,
    estimators: pd.DataFrame,
    obs_group_joinids: pd.DataFrame,
) -> Tuple[np.float32, np.float32, np.float32]:
    # ensure estimators are available for all obs groups (for when feature had no expression data for some obs groups)
    estimators = fill_missing_data(obs_group_joinids, estimators)

    assert len(estimators) == len(design)

    # Transform to log space (alternatively can resample in log space)
    lm, selm = transform_to_log_space(estimators["mean"].to_numpy(), estimators["sem"].to_numpy())

    return de_wls(X=design.values, y=lm, n=cell_counts, v=selm**2)


# @timeit
def transform_to_log_space(
    m: npt.NDArray[np.float32], sem: npt.NDArray[np.float32]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    lm = np.log(m)
    selm = (np.log(m + sem) - np.log(m - sem)) / 2
    assert (selm > 0).all()
    return lm, selm


# @timeit
def fill_missing_data(obs_group_joinids: pd.DataFrame, feature_estimators: pl.DataFrame) -> pl.DataFrame:
    feature_estimators = pl.DataFrame(obs_group_joinids).join(
        feature_estimators[["obs_group_joinid", "mean", "sem"]], on="obs_group_joinid", how="left"
    )

    return feature_estimators.with_columns(
        feature_estimators["mean"].fill_null(1e-3), feature_estimators["sem"].fill_null(1e-4)
    )


# @timeit
def de_wls_fit(X: npt.NDArray[np.float32], y: npt.NDArray[np.float32], n: npt.NDArray[np.float32]) -> np.float32:
    # fit WLS using sample_weights
    WLS = LinearRegression()
    WLS.fit(X, y, sample_weight=n)

    # note: we have all the other coefficients (i.e. effect size) for the other covariates here as well, but we only
    # want the treatment effect for now
    return cast(np.float32, WLS.coef_[0])


# @timeit
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


# @timeit
def de_wls_stats_pinv(m: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.linalg.pinv(m)


# @timeit
def de_wls_stats_matmul(W: npt.NDArray[np.float32], X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return (X.T * W) @ X


# @timeit
def de_wls_stats_W(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 1 / v


# @timeit
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
        print("Usage: python diff_expr.py <filter> <treatment> <cube_path> <n_processes> <covariates>")
        sys.exit(1)

    filter_arg, treatment_arg, cube_path_arg, n_processes, covariates = sys.argv[1:6]

    logging.getLogger().setLevel(logging.DEBUG)

    de_result = compute_all(
        cube_path_arg, filter_arg, treatment_arg, int(n_processes), covariates.split(",") if covariates else None
    )

    # Output DE result
    print(de_result[0])
