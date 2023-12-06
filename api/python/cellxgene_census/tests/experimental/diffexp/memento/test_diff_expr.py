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

    for c, estimator in enumerate(["nnz", "n_obs", "min", "max", "sum", "mean", "sem", "var", "sev", "selv"], start=1):
        estimators[estimator] = range(1, len(estimators) + 1)
        estimators[estimator] *= c

    return estimators


@pytest.mark.parametrize("dim_counts", [{"feature_id": 3, "cell_type": 3}])
def test__setup__too_many_treatment_values__fails(estimators_df: pd.DataFrame) -> None:
    with pytest.raises(AssertionError, match="treatment must have exactly 2 distinct values"):
        diff_expr.setup(estimators_df, "cell_type")


@pytest.mark.parametrize("dim_counts", [{"feature_id": 3, "cell_type": 2, "dataset_id": 3, "assay": 2}])
def test_setup(estimators_df: pd.DataFrame) -> None:
    cell_counts, design, features, mean, se_mean = diff_expr.setup(estimators_df, "cell_type")

    assert list(cell_counts) == list(range(2, 73, 6))
    assert features == ["feature_id_a", "feature_id_b", "feature_id_c"]
    print(design.to_dict(orient="records"))
    print(mean.to_dict(orient="records"))
    print(se_mean.to_dict(orient="records"))
    assert design.to_dict(orient="records") == [
        {"cell_type_cell_type_b": 0, "dataset_id_dataset_id_b": 0, "dataset_id_dataset_id_c": 0, "assay_assay_b": 0},
        {"cell_type_cell_type_b": 0, "dataset_id_dataset_id_b": 0, "dataset_id_dataset_id_c": 0, "assay_assay_b": 1},
        {"cell_type_cell_type_b": 0, "dataset_id_dataset_id_b": 1, "dataset_id_dataset_id_c": 0, "assay_assay_b": 0},
        {"cell_type_cell_type_b": 0, "dataset_id_dataset_id_b": 1, "dataset_id_dataset_id_c": 0, "assay_assay_b": 1},
        {"cell_type_cell_type_b": 0, "dataset_id_dataset_id_b": 0, "dataset_id_dataset_id_c": 1, "assay_assay_b": 0},
        {"cell_type_cell_type_b": 0, "dataset_id_dataset_id_b": 0, "dataset_id_dataset_id_c": 1, "assay_assay_b": 1},
        {"cell_type_cell_type_b": 1, "dataset_id_dataset_id_b": 0, "dataset_id_dataset_id_c": 0, "assay_assay_b": 0},
        {"cell_type_cell_type_b": 1, "dataset_id_dataset_id_b": 0, "dataset_id_dataset_id_c": 0, "assay_assay_b": 1},
        {"cell_type_cell_type_b": 1, "dataset_id_dataset_id_b": 1, "dataset_id_dataset_id_c": 0, "assay_assay_b": 0},
        {"cell_type_cell_type_b": 1, "dataset_id_dataset_id_b": 1, "dataset_id_dataset_id_c": 0, "assay_assay_b": 1},
        {"cell_type_cell_type_b": 1, "dataset_id_dataset_id_b": 0, "dataset_id_dataset_id_c": 1, "assay_assay_b": 0},
        {"cell_type_cell_type_b": 1, "dataset_id_dataset_id_b": 0, "dataset_id_dataset_id_c": 1, "assay_assay_b": 1},
    ]

    assert mean.to_dict(orient="records") == [
        {"feature_id_a": 6, "feature_id_b": 12, "feature_id_c": 18},
        {"feature_id_a": 24, "feature_id_b": 30, "feature_id_c": 36},
        {"feature_id_a": 42, "feature_id_b": 48, "feature_id_c": 54},
        {"feature_id_a": 60, "feature_id_b": 66, "feature_id_c": 72},
        {"feature_id_a": 78, "feature_id_b": 84, "feature_id_c": 90},
        {"feature_id_a": 96, "feature_id_b": 102, "feature_id_c": 108},
        {"feature_id_a": 114, "feature_id_b": 120, "feature_id_c": 126},
        {"feature_id_a": 132, "feature_id_b": 138, "feature_id_c": 144},
        {"feature_id_a": 150, "feature_id_b": 156, "feature_id_c": 162},
        {"feature_id_a": 168, "feature_id_b": 174, "feature_id_c": 180},
        {"feature_id_a": 186, "feature_id_b": 192, "feature_id_c": 198},
        {"feature_id_a": 204, "feature_id_b": 210, "feature_id_c": 216},
    ]

    assert se_mean.to_dict(orient="records") == [
        {"feature_id_a": 7, "feature_id_b": 14, "feature_id_c": 21},
        {"feature_id_a": 28, "feature_id_b": 35, "feature_id_c": 42},
        {"feature_id_a": 49, "feature_id_b": 56, "feature_id_c": 63},
        {"feature_id_a": 70, "feature_id_b": 77, "feature_id_c": 84},
        {"feature_id_a": 91, "feature_id_b": 98, "feature_id_c": 105},
        {"feature_id_a": 112, "feature_id_b": 119, "feature_id_c": 126},
        {"feature_id_a": 133, "feature_id_b": 140, "feature_id_c": 147},
        {"feature_id_a": 154, "feature_id_b": 161, "feature_id_c": 168},
        {"feature_id_a": 175, "feature_id_b": 182, "feature_id_c": 189},
        {"feature_id_a": 196, "feature_id_b": 203, "feature_id_c": 210},
        {"feature_id_a": 217, "feature_id_b": 224, "feature_id_c": 231},
        {"feature_id_a": 238, "feature_id_b": 245, "feature_id_c": 252},
    ]
