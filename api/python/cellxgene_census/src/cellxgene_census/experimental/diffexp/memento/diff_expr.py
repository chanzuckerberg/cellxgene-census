#!/usr/bin/env python
import itertools
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats
import tiledb

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


def query_estimators(cube_path: str, obs_groups_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    with tiledb.open(os.path.join(cube_path, ESTIMATORS_ARRAY), "r") as estimators_array:
        estimators_df = estimators_array.df[features, obs_groups_df.obs_group_joinid.values]

        # TODO: Determine whether it's reasonable to drop these values, or if we should revisit how they're being
        #  computed in the first place. If reasonable, this filtering should be done by the cube builder, not here.
        # This filtering ensures that we will  not take of logs of non-positive values, or end up with selm values
        # of 0
        drop_mask = (estimators_df["sem"] <= 0) | (estimators_df["sem"] >= estimators_df["mean"])
        if drop_mask.any():
            logging.warning(
                f"dropping {drop_mask.sum()} rows with invalid values ({drop_mask.sum() / len(drop_mask):.2%})"
            )
            estimators_df = estimators_df[~drop_mask]

        return obs_groups_df.merge(estimators_df, on="obs_group_joinid")


def compute_hypothesis_test(cube_path: str, query_filter: str, treatment: str, n_threads: int) -> pd.DataFrame:
    with tiledb.open(os.path.join(cube_path, OBS_GROUPS_ARRAY), "r") as obs_groups_array:
        obs_groups_df = obs_groups_array.query(cond=query_filter or None).df[:]

        distinct_treatment_values = obs_groups_array.query().df[:][treatment].nunique()
        assert distinct_treatment_values == 2, "treatment must have exactly 2 distinct values"

        # convert categorical columns to ints
        for col in obs_groups_df.select_dtypes(include=["category"]).columns:
            obs_groups_df[col] = obs_groups_df[col].cat.codes

    # TODO: need canonical list of features efficiently
    with tiledb.open(os.path.join(cube_path, ESTIMATORS_ARRAY), "r") as estimators_array:
        features = estimators_array.query(attrs=[], dims=["feature_id"]).df[:]["feature_id"].drop_duplicates().tolist()

    # partition features into N groups
    feature_groups = [features.tolist() for features in np.split(np.array(features), min(len(features), n_threads))]

    # compute each feature group in parallel
    result_groups = ProcessPoolExecutor(max_workers=n_threads).map(
        compute_hypothesis_test_features,
        itertools.cycle([cube_path]),
        itertools.cycle([obs_groups_df]),
        itertools.cycle([treatment]),
        feature_groups,
    )

    # flatten results
    results = itertools.chain.from_iterable(result_groups)

    return pd.DataFrame(results, columns=["feature_id", "coef", "z", "pval"], copy=False).set_index("feature_id")


def compute_hypothesis_test_features(
    cube_path: str, obs_groups_df: pd.DataFrame, treatment: str, features: List[str]
) -> List[Tuple[str, np.float32, np.float32, np.float32]]:
    cell_counts, design, mean, se_mean = load_data(cube_path, obs_groups_df, treatment, features)
    return [
        (feature, *compute_hypothesis_test_feature(cell_counts, design, mean, se_mean, feature))  # type:ignore
        for feature in features
    ]


def compute_hypothesis_test_feature(
    cell_counts: npt.NDArray[np.float32],
    design: pd.DataFrame,
    mean: pd.DataFrame,
    se_mean: pd.DataFrame,
    feature: str,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    m = cast(npt.NDArray[np.float32], mean[feature].values)
    sem = cast(npt.NDArray[np.float32], se_mean[feature].values)
    # Transform to log space (alternatively can resample in log space)
    lm = np.log(m)
    selm = (np.log(m + sem) - np.log(m - sem)) / 2
    assert (selm > 0).all()
    return de_wls(X=design.values, y=lm, n=cell_counts, v=selm**2)


def load_data(
    cube_path: str, obs_groups_df: pd.DataFrame, treatment: str, features: List[str]
) -> Tuple[npt.NDArray[np.float32], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cube = query_estimators(cube_path, obs_groups_df, features)

    # make treatment variable be in the first column of the design matrix
    variables = [treatment] + [covariate for covariate in CUBE_LOGICAL_DIMS_OBS if covariate != treatment]
    # make a table with a column per feature, and a row per obs group
    mean = cube.pivot_table(index=variables, columns="feature_id", values="mean").fillna(1e-3)
    se_mean = cube.pivot_table(index=variables, columns="feature_id", values="sem").fillna(1e-4)

    # TODO: "n_obs" can be stored on obs_groups array, and so cell_counts can be computed from that instead of `cube`
    groups = cube[variables + ["n_obs"]].drop_duplicates(variables)
    cell_counts = cast(npt.NDArray[np.float32], groups["n_obs"].values)
    design = pd.get_dummies(groups[variables], drop_first=True, dtype=int)

    n_groups = len(groups)
    n_features = cube["feature_id"].nunique()
    assert len(cell_counts) == n_groups
    assert design.shape[0] == n_groups
    assert mean.shape == (n_groups, n_features)
    assert se_mean.shape == (n_groups, n_features)

    return cell_counts, design, mean, se_mean


def de_wls(
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    n: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Perform DE for each gene using Weighted Least Squares (i.e., a weighted Linear Regression model)
    """

    from sklearn.linear_model import LinearRegression

    # fit WLS using sample_weights
    WLS = LinearRegression()
    WLS.fit(X, y, sample_weight=n)

    # note: we have all the other coefficients (i.e. effect size) for the other covariates here as well, but we only
    # want the treatment effect for now
    treatment_col = 0
    coef = WLS.coef_[treatment_col]

    W = np.diag(1 / v)

    beta_var_hat = np.diag(np.linalg.pinv(X.T @ W @ X))
    se = np.sqrt(beta_var_hat[treatment_col])

    z = coef / se
    pv = stats.norm.sf(np.abs(z)) * 2

    return coef, z, pv


# Script entrypoint
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python diff_expr.py <filter> <treatment> <cube_path> <csv_output_path>")
        sys.exit(1)

    filter_arg, treatment_arg, cube_path_arg, csv_output_path_arg = sys.argv[1:5]

    de_result = compute_hypothesis_test(cube_path_arg, filter_arg, treatment_arg, n_threads=os.cpu_count() or 1)

    # Output DE result
    print(de_result)
    de_result.to_csv(csv_output_path_arg)
