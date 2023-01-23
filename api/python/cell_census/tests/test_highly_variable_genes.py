import pytest
import tiledbsoma as soma

import cell_census
from cell_census.compute import highly_variable_genes


@pytest.mark.live_corpus
def test_highly_variable_genes() -> None:
    census = cell_census.open_soma(census_version="latest")
    experiment = census["census_data"]["homo_sapiens"]

    query = soma.ExperimentAxisQuery(
        experiment,
        "RNA",
        obs_query=soma.AxisQuery(value_filter="tissue_general == 'scalp' and is_primary_data == True"),
        var_query=soma.AxisQuery(coords=(slice(0, 31),)),
    )

    result = highly_variable_genes(query, n_top_genes=10)

    assert result.shape == (32, 5)
    assert list(result.columns) == ["means", "variances", "highly_variable_rank", "variances_norm", "highly_variable"]
    assert result[result["highly_variable"]].shape[0] == 10
    # TODO: assert the computed highly variable genes are in fact the correct ones
