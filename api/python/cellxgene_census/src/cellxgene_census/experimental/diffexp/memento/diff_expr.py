#!/usr/bin/env python
import os
import sys
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


def run(cube_path_: str, filter_: str, treatment: str) -> pd.DataFrame:
    cube_df = query_estimators(cube_path_, filter_)
    cell_counts, design, features, mean, se_mean = setup(cube_df, treatment)
    return compute_hypothesis_test(cell_counts, design, features, mean, se_mean)


def query_estimators(cube_path_: str, filter_: str) -> pd.DataFrame:
    with tiledb.open(os.path.join(cube_path_, OBS_GROUPS_ARRAY), "r") as obs_groups_array:
        with tiledb.open(os.path.join(cube_path_, ESTIMATORS_ARRAY), "r") as estimators_array:
            obs_groups_df = obs_groups_array.query(cond=filter_ or None).df[:]
            # convert categorical columns to ints
            for col in obs_groups_df.select_dtypes(include=["category"]).columns:
                obs_groups_df[col] = obs_groups_df[col].cat.codes
            estimators_df = estimators_array.df[obs_groups_df.obs_group_joinid.values]
            return cast(pd.DataFrame, obs_groups_df.merge(estimators_df, on="obs_group_joinid"))


def compute_hypothesis_test(
    cell_counts: npt.NDArray[np.float64],
    design: pd.DataFrame,
    features: List[str],
    mean: pd.DataFrame,
    se_mean: pd.DataFrame,
) -> pd.DataFrame:
    de_result = []
    # TODO: parallelize
    for feature in features:
        m = cast(npt.NDArray[np.float64], mean[feature].values)
        sem = cast(npt.NDArray[np.float64], se_mean[feature].values)

        # Transform to log space (alternatively can resample in log space)
        lm = np.log(m)
        selm = (np.log(m + sem) - np.log(m - sem)) / 2

        coef, z, pv = de_wls(X=design.values, y=lm, n=cell_counts, v=selm**2)
        de_result.append((feature, coef, z, pv))

    return pd.DataFrame(de_result, columns=["feature_id", "coef", "z", "pval"]).set_index("feature_id")


def setup(
    cube: pd.DataFrame, treatment_variable: str
) -> Tuple[npt.NDArray[np.float64], pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
    distinct_treatment_values = cube[[treatment_variable]].nunique()[0]
    assert distinct_treatment_values == 2, "treatment must have exactly 2 distinct values"

    # make treatment variable be in the first column of the design matrix
    variables = [treatment_variable] + [
        covariate for covariate in CUBE_LOGICAL_DIMS_OBS if covariate != treatment_variable
    ]

    mean = cube.pivot_table(index=variables, columns="feature_id", values="mean").fillna(1e-3)
    se_mean = cube.pivot_table(index=variables, columns="feature_id", values="sem").fillna(1e-4)

    groups = cube[variables + ["n_obs"]].drop_duplicates(variables)
    cell_counts = cast(npt.NDArray[np.float64], groups["n_obs"].values)
    design = pd.get_dummies(groups[variables], drop_first=True, dtype=int)

    features = cube["feature_id"].drop_duplicates().tolist()

    assert len(cell_counts) == len(groups)
    assert design.shape[0] == len(groups)
    assert mean.shape == (len(groups), len(features))
    assert se_mean.shape == (len(groups), len(features))

    return cell_counts, design, features, mean, se_mean


def de_wls(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    de_result = run(cube_path_arg, filter_arg, treatment_arg)

    # Output DE result
    print(de_result)
    de_result.to_csv(csv_output_path_arg)
