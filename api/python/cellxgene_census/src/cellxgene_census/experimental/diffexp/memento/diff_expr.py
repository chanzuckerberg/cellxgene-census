#!/usr/bin/env python
import itertools
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats
import tiledb
from sklearn.linear_model import LinearRegression

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
    tiledb_config = {
        "soma.init_buffer_bytes": 2**31,
    }
    with tiledb.open(os.path.join(cube_path, ESTIMATORS_ARRAY), "r", config=tiledb_config) as estimators_array:
        estimators_df = estimators_array.df[features, obs_groups_df.obs_group_joinid.values]
        # TODO: Determine whether it's reasonable to drop these values, or if we should revisit how they're being
        #  computed in the first place. If reasonable, this filtering should be done by the cube builder, not here.
        # This filtering ensures that we will not take of logs of non-positive values, or end up with selm values of 0
        drop_mask = (estimators_df["sem"] <= 0) | (estimators_df["sem"] >= estimators_df["mean"])
        if drop_mask.any():
            logging.warning(
                f"dropping {drop_mask.sum()} rows with invalid values ({drop_mask.sum() / len(drop_mask):.2%})"
            )
            estimators_df = estimators_df[~drop_mask]

        return cast(pd.DataFrame, estimators_df)


def compute_all(cube_path: str, query_filter: str, treatment: str, n_threads: int) -> pd.DataFrame:
    with tiledb.open(os.path.join(cube_path, OBS_GROUPS_ARRAY), "r") as obs_groups_array:
        obs_groups_df = obs_groups_array.query(cond=query_filter or None).df[:]

        distinct_treatment_values = obs_groups_df[treatment].nunique()
        assert distinct_treatment_values == 2, "treatment must have exactly 2 distinct values"

        # convert categorical columns to ints
        for col in obs_groups_df.select_dtypes(include=["category"]).columns:
            obs_groups_df[col] = obs_groups_df[col].cat.codes

    # TODO: need canonical list of features efficiently
    features = get_features(cube_path)[:100]

    # compute each feature group in parallel
    n_feature_groups = min(len(features), n_threads)
    feature_groups = [features.tolist() for features in np.array_split(np.array(features), n_feature_groups)]
    print(f"computing for {len(obs_groups_df)} obs groups ({obs_groups_df.n_obs.sum()} cells) and {len(features)} features using {n_feature_groups} processes, {len(features) // n_feature_groups} features/process")

    # make treatment variable be in the first column of the design matrix
    variables = [treatment] + [covariate for covariate in CUBE_LOGICAL_DIMS_OBS if covariate != treatment]
    design = pd.get_dummies(obs_groups_df[variables], drop_first=True, dtype=int)

    result_groups = ProcessPoolExecutor(max_workers=n_threads).map(
        partial(compute_for_features, cube_path, design, obs_groups_df),
        feature_groups,
        range(len(feature_groups))
    )

    # flatten results
    results = itertools.chain.from_iterable(result_groups)

    return pd.DataFrame(results, columns=["feature_id", "coef", "z", "pval"], copy=False).set_index("feature_id")


def get_features(cube_path):
    feature_id_path = os.path.join(cube_path, "feature_ids.json")
    if os.path.isfile(feature_id_path):
        with open(feature_id_path) as f:
            features = json.load(f)
    else:
        with tiledb.open(os.path.join(cube_path, ESTIMATORS_ARRAY), "r",
                         config={"soma.init_buffer_bytes": 2 ** 32}) as estimators_array:
            features = estimators_array.query(attrs=[], dims=["feature_id"]).df[:][
                "feature_id"].drop_duplicates().tolist()
            with open(feature_id_path, "w") as f:
                json.dump(features, f)
    return features


def compute_for_features(cube_path: str, design: pd.DataFrame, obs_groups_df: pd.DataFrame, features: List[str],
                         feature_group_key: int) -> List[Tuple[str, np.float32, np.float32, np.float32]]:
    print(f"computing for feature group {feature_group_key}, {features[0]}..{features[-1]}...")
    estimators = query_estimators(cube_path, obs_groups_df, features)
    cell_counts = obs_groups_df["n_obs"].values
    obs_group_joinids = obs_groups_df[["obs_group_joinid"]]

    result = [
        (feature, *compute_for_feature(cell_counts, obs_group_joinids, design, estimators, feature))
        for feature in features
    ]

    print(f"computed for feature group {feature_group_key}, {features[0]}..{features[-1]}")

    return result  # type:ignore


def compute_for_feature(
    cell_counts: npt.NDArray[np.float32],
    obs_group_joinids: pd.DataFrame,
    design: pd.DataFrame,
    estimators: pd.DataFrame,
    feature: str,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    # extract estimators for the specified feature
    feature_estimators = estimators[estimators.feature_id == feature][["obs_group_joinid", "mean", "sem"]]

    # ensure estimators are available for all obs groups (for when feature had no expression data for some obs groups)
    feature_estimators = obs_group_joinids.merge(
        feature_estimators, on="obs_group_joinid", how="left"
    )
    m = cast(npt.NDArray[np.float32], feature_estimators["mean"].fillna(1e-3).values)
    sem = cast(npt.NDArray[np.float32], feature_estimators["sem"].fillna(1e-4).values)

    assert len(m) == len(design)
    assert len(sem) == len(design)

    # Transform to log space (alternatively can resample in log space)
    lm = np.log(m)
    selm = (np.log(m + sem) - np.log(m - sem)) / 2
    assert (selm > 0).all()

    return de_wls(X=design.values, y=lm, n=cell_counts, v=selm**2)


def de_wls(
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    n: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Perform DE for each gene using Weighted Least Squares (i.e., a weighted Linear Regression model)
    """

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

    de_result = compute_all(cube_path_arg, filter_arg, treatment_arg, os.cpu_count() or 1)

    # Output DE result
    print(de_result)
    de_result.to_csv(csv_output_path_arg)
