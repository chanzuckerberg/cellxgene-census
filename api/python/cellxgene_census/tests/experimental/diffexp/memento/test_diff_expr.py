from os import path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from cellxgene_census.experimental.diffexp.memento import diff_expr
from cellxgene_census.experimental.diffexp.memento.diff_expr import CUBE_LOGICAL_DIMS_OBS


class TestDiffExprRealDataset:
    """
    This class contains regression tests that run on realistic datasets.

    Since this class is intended to be serve as a regression test suite
    using real data, it is strongly recommended that the functions under
    test are all PUBLIC functions. Private functions are best encapsulated
    in separate class.
    """

    @pytest.fixture(scope="class", params=["test_case_1", "test_case_2"])
    def test_cases_for_compute_all_fn(self, request: Any) -> Dict[str, Any]:
        """
        Fixture that generates test cases for function
        calls to `compute_all` given a test case name.

        This fixture returns a tuple `(diff_exp_query, expected_result)`
        such that `diff_exp_query` encapsulates the differential expression
        query that will be executed by `compute_all()`.

        `expected_result` is a datastructure containing the pertinent parts of
        the return value of `compute_all()`.
        """
        # TODO: Figure out a common location to store estimator cube fixtures so that it is
        # explicitly clear that both the differential expression API and differential expression cube builder
        # components use it for testing
        pwd = path.dirname(__file__)
        estimator_cube_path = path.join(
            pwd, "../../../../../../../tools/models/memento/tests/fixtures/estimators-cube-expected/"
        )

        test_cases = {
            "test_case_1": {
                "diff_exp_query": {
                    "cube_path": estimator_cube_path,
                    "query_filter": "tissue_general_ontology_term_id in ['UBERON:0001723'] and sex_ontology_term_id in ['PATO:0000383', 'PATO:0000384']",
                    "treatment": "sex_ontology_term_id",
                    "num_sampled_genes": 2,
                },
                "expected_diff_exp_result": [
                    ("ENSG00000000419", -0.111612, -1.895204, 0.058065),
                    ("ENSG00000002330", 0.229054, 4.085651, 0.000044),
                ],
            },
            "test_case_2": {
                "diff_exp_query": {
                    "cube_path": estimator_cube_path,
                    "query_filter": "tissue_general_ontology_term_id in ['UBERON:0001723'] and cell_type_ontology_term_id in ['CL:0000066', 'CL:0000057']",
                    "treatment": "cell_type_ontology_term_id",
                    "num_sampled_genes": 2,
                },
                "expected_diff_exp_result": [
                    ("ENSG00000000419", 0.868715, 6.048411, 1.462810e-09),
                    ("ENSG00000002330", 0.834346, 5.218739, 1.801458e-07),
                ],
            },
        }

        return test_cases[request.param]

    def test_diff_exp_query_basic(self, test_cases_for_compute_all_fn: Any) -> None:
        # Arrange
        estimator_cube_path = test_cases_for_compute_all_fn["diff_exp_query"]["cube_path"]
        query_filter = test_cases_for_compute_all_fn["diff_exp_query"]["query_filter"]
        treatment = test_cases_for_compute_all_fn["diff_exp_query"]["treatment"]
        num_sampled_genes = test_cases_for_compute_all_fn["diff_exp_query"]["num_sampled_genes"]

        # Act
        observed_diff_exp_result_df, _ = diff_expr.compute_all(
            cube_path=estimator_cube_path,
            query_filter=query_filter,
            treatment=treatment,
            n_features=num_sampled_genes,
            n_processes=1,
        )

        observed_diff_exp_result_df = observed_diff_exp_result_df.reset_index().set_index("feature_id").sort_index()

        expected_data = test_cases_for_compute_all_fn["expected_diff_exp_result"]
        expected_diff_exp_result_df = (
            pd.DataFrame(expected_data, columns=["feature_id", "coef", "z", "pval"])
            .set_index("feature_id")
            .sort_index()
        )

        # Assert
        assert np.allclose(observed_diff_exp_result_df.values, expected_diff_exp_result_df.values, atol=1e-07)
