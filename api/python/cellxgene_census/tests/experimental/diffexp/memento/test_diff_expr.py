import itertools
from typing import Dict

import pandas as pd
import pytest

from cellxgene_census.experimental.diffexp.memento import diff_expr
from cellxgene_census.experimental.diffexp.memento.diff_expr import CUBE_LOGICAL_DIMS_OBS


@pytest.fixture(scope="function")
def estimators_df(dim_counts: Dict[str, int]) -> pd.DataFrame:
    """Create a dummy estimators DataFrame, with all combinations of variables."""
    columns = CUBE_LOGICAL_DIMS_OBS + ["feature_id"]
    dim_counts_full = {var: 1 for var in columns}
    dim_counts_full.update(dim_counts or {})
    dim_values = {var: [f"{var}_{chr(ord('a') + i)}" for i in range(n)] for var, n in dim_counts_full.items()}

    rows = list(itertools.product(*dim_values.values()))
    estimators = pd.DataFrame(data=rows, columns=columns).astype("str")

    for c, estimator in enumerate(["n_obs", "mean", "sem", "var", "selv"], start=1):
        estimators[estimator] = range(1, len(estimators) + 1)
        estimators[estimator] *= c

    return estimators


@pytest.mark.parametrize("dim_counts", [{"feature_id": 3, "cell_type_ontology_term_id": 3}])
def test__setup__too_many_treatment_values__fails(estimators_df: pd.DataFrame) -> None:
    with pytest.raises(AssertionError, match="treatment must have exactly 2 distinct values"):
        diff_expr.setup(estimators_df, "cell_type_ontology_term_id")


@pytest.mark.parametrize(
    "dim_counts", [{"feature_id": 3, "cell_type_ontology_term_id": 2, "dataset_id": 3, "assay_ontology_term_id": 2}]
)
def test_setup(estimators_df: pd.DataFrame) -> None:
    cell_counts, design, features, mean, se_mean = diff_expr.setup(estimators_df, "cell_type_ontology_term_id")

    # Note: Uncomment below code block to retrieve new expected values, if test data changes.
    # Manually verify before replacing expected values below!

    # print(list(cell_counts))
    # print(features)
    # print(design.to_dict(orient="records"))
    # print(mean.to_dict(orient="records"))
    # print(se_mean.to_dict(orient="records"))

    assert list(cell_counts) == [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
    assert features == ["feature_id_a", "feature_id_b", "feature_id_c"]
    assert design.to_dict(orient="records") == [
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 0,
            "dataset_id_dataset_id_b": 0,
            "dataset_id_dataset_id_c": 0,
            "assay_ontology_term_id_assay_ontology_term_id_b": 0,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 0,
            "dataset_id_dataset_id_b": 0,
            "dataset_id_dataset_id_c": 0,
            "assay_ontology_term_id_assay_ontology_term_id_b": 1,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 0,
            "dataset_id_dataset_id_b": 1,
            "dataset_id_dataset_id_c": 0,
            "assay_ontology_term_id_assay_ontology_term_id_b": 0,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 0,
            "dataset_id_dataset_id_b": 1,
            "dataset_id_dataset_id_c": 0,
            "assay_ontology_term_id_assay_ontology_term_id_b": 1,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 0,
            "dataset_id_dataset_id_b": 0,
            "dataset_id_dataset_id_c": 1,
            "assay_ontology_term_id_assay_ontology_term_id_b": 0,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 0,
            "dataset_id_dataset_id_b": 0,
            "dataset_id_dataset_id_c": 1,
            "assay_ontology_term_id_assay_ontology_term_id_b": 1,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 1,
            "dataset_id_dataset_id_b": 0,
            "dataset_id_dataset_id_c": 0,
            "assay_ontology_term_id_assay_ontology_term_id_b": 0,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 1,
            "dataset_id_dataset_id_b": 0,
            "dataset_id_dataset_id_c": 0,
            "assay_ontology_term_id_assay_ontology_term_id_b": 1,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 1,
            "dataset_id_dataset_id_b": 1,
            "dataset_id_dataset_id_c": 0,
            "assay_ontology_term_id_assay_ontology_term_id_b": 0,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 1,
            "dataset_id_dataset_id_b": 1,
            "dataset_id_dataset_id_c": 0,
            "assay_ontology_term_id_assay_ontology_term_id_b": 1,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 1,
            "dataset_id_dataset_id_b": 0,
            "dataset_id_dataset_id_c": 1,
            "assay_ontology_term_id_assay_ontology_term_id_b": 0,
        },
        {
            "cell_type_ontology_term_id_cell_type_ontology_term_id_b": 1,
            "dataset_id_dataset_id_b": 0,
            "dataset_id_dataset_id_c": 1,
            "assay_ontology_term_id_assay_ontology_term_id_b": 1,
        },
    ]
    assert mean.to_dict(orient="records") == [
        {"feature_id_a": 2.0, "feature_id_b": 4.0, "feature_id_c": 6.0},
        {"feature_id_a": 8.0, "feature_id_b": 10.0, "feature_id_c": 12.0},
        {"feature_id_a": 14.0, "feature_id_b": 16.0, "feature_id_c": 18.0},
        {"feature_id_a": 20.0, "feature_id_b": 22.0, "feature_id_c": 24.0},
        {"feature_id_a": 26.0, "feature_id_b": 28.0, "feature_id_c": 30.0},
        {"feature_id_a": 32.0, "feature_id_b": 34.0, "feature_id_c": 36.0},
        {"feature_id_a": 38.0, "feature_id_b": 40.0, "feature_id_c": 42.0},
        {"feature_id_a": 44.0, "feature_id_b": 46.0, "feature_id_c": 48.0},
        {"feature_id_a": 50.0, "feature_id_b": 52.0, "feature_id_c": 54.0},
        {"feature_id_a": 56.0, "feature_id_b": 58.0, "feature_id_c": 60.0},
        {"feature_id_a": 62.0, "feature_id_b": 64.0, "feature_id_c": 66.0},
        {"feature_id_a": 68.0, "feature_id_b": 70.0, "feature_id_c": 72.0},
    ]
    assert se_mean.to_dict(orient="records") == [
        {"feature_id_a": 3.0, "feature_id_b": 6.0, "feature_id_c": 9.0},
        {"feature_id_a": 12.0, "feature_id_b": 15.0, "feature_id_c": 18.0},
        {"feature_id_a": 21.0, "feature_id_b": 24.0, "feature_id_c": 27.0},
        {"feature_id_a": 30.0, "feature_id_b": 33.0, "feature_id_c": 36.0},
        {"feature_id_a": 39.0, "feature_id_b": 42.0, "feature_id_c": 45.0},
        {"feature_id_a": 48.0, "feature_id_b": 51.0, "feature_id_c": 54.0},
        {"feature_id_a": 57.0, "feature_id_b": 60.0, "feature_id_c": 63.0},
        {"feature_id_a": 66.0, "feature_id_b": 69.0, "feature_id_c": 72.0},
        {"feature_id_a": 75.0, "feature_id_b": 78.0, "feature_id_c": 81.0},
        {"feature_id_a": 84.0, "feature_id_b": 87.0, "feature_id_c": 90.0},
        {"feature_id_a": 93.0, "feature_id_b": 96.0, "feature_id_c": 99.0},
        {"feature_id_a": 102.0, "feature_id_b": 105.0, "feature_id_c": 108.0},
    ]
