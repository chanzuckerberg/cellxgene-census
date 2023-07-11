import math

import pandas as pd
import pytest
import tiledbsoma as soma

import cellxgene_census
from cellxgene_census.experimental import pp


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize(
    "experiment_name,obs_value_filter",
    [
        ("mus_musculus", 'tissue_general == "liver" and is_primary_data == True'),
        ("mus_musculus", 'is_primary_data == True and tissue_general == "heart"'),
        pytest.param("mus_musculus", "is_primary_data == True", marks=pytest.mark.expensive),
        pytest.param("homo_sapiens", "is_primary_data == True", marks=pytest.mark.expensive),
    ],
)
def test_mean_variance(
    experiment_name: str,
    obs_value_filter: str,
    small_mem_context: soma.SOMATileDBContext,
) -> None:
    with cellxgene_census.open_soma(census_version="latest", context=small_mem_context) as census:
        with census["census_data"][experiment_name].axis_query(
            measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
        ) as query:
            mean_variance = pp.mean_variance(query, calculate_mean=True, calculate_variance=True)
            assert isinstance(mean_variance, pd.DataFrame)
            assert "mean" in mean_variance
            assert "variance" in mean_variance
            assert mean_variance.index.name == "soma_joinid"

            # Pick one element from the dataframe
            test_soma_joinid = int(mean_variance.index[0])  # type: ignore

            with census["census_data"][experiment_name].axis_query(
                measurement_name="RNA", obs_query=soma.AxisQuery(coords=([test_soma_joinid],))
            ) as row_query:
                sparse_row = row_query.X("raw").coos((1 + test_soma_joinid, row_query.n_vars)).concat()
                sparse_mat = sparse_row.to_scipy()
                row = sparse_mat.tocsr().getrow(test_soma_joinid)

                assert math.isclose(
                    row.todense().mean(), mean_variance.loc[test_soma_joinid, "mean"], rel_tol=0.01  # type: ignore
                )
                assert math.isclose(
                    row.todense().var(), mean_variance.loc[test_soma_joinid, "variance"], rel_tol=0.01  # type: ignore
                )


def test_mean_variance_no_flags() -> None:
    with pytest.raises(ValueError):
        pp.mean_variance(None, calculate_mean=False, calculate_variance=False)


def test_mean_variance_empty_query() -> None:
    with pytest.raises(ValueError):
        pp.mean_variance(soma.AxisQuery(), calculate_mean=True, calculate_variance=True)


def test_mean_variance_wrong_layer() -> None:
    with pytest.raises(ValueError):
        pp.mean_variance(soma.AxisQuery(), calculate_mean=True, calculate_variance=True, layer="foo")


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize(
    "experiment_name,obs_value_filter",
    [
        ("mus_musculus", 'tissue_general == "liver" and is_primary_data == True'),
        ("mus_musculus", 'is_primary_data == True and tissue_general == "heart"'),
        pytest.param("mus_musculus", "is_primary_data == True", marks=pytest.mark.expensive),
        pytest.param("homo_sapiens", "is_primary_data == True", marks=pytest.mark.expensive),
    ],
)
def test_mean(
    experiment_name: str,
    obs_value_filter: str,
    small_mem_context: soma.SOMATileDBContext,
) -> None:
    with cellxgene_census.open_soma(census_version="latest", context=small_mem_context) as census:
        with census["census_data"][experiment_name].axis_query(
            measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
        ) as query:
            mean_variance = pp.mean_variance(query, calculate_mean=True, calculate_variance=False)
            assert isinstance(mean_variance, pd.DataFrame)
            assert "mean" in mean_variance
            assert "variance" not in mean_variance
            assert mean_variance.index.name == "soma_joinid"

            # Pick one element from the dataframe
            test_soma_joinid = int(mean_variance.index[0])  # type: ignore

            with census["census_data"][experiment_name].axis_query(
                measurement_name="RNA", obs_query=soma.AxisQuery(coords=([test_soma_joinid],))
            ) as row_query:
                sparse_row = row_query.X("raw").coos((1 + test_soma_joinid, row_query.n_vars)).concat()
                sparse_mat = sparse_row.to_scipy()
                row = sparse_mat.tocsr().getrow(test_soma_joinid)

                assert math.isclose(
                    row.todense().mean(), mean_variance.loc[test_soma_joinid, "mean"], rel_tol=0.01  # type: ignore
                )


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize(
    "experiment_name,obs_value_filter",
    [
        ("mus_musculus", 'tissue_general == "liver" and is_primary_data == True'),
        ("mus_musculus", 'is_primary_data == True and tissue_general == "heart"'),
    ],
)
def test_variance(
    experiment_name: str,
    obs_value_filter: str,
    small_mem_context: soma.SOMATileDBContext,
) -> None:
    with cellxgene_census.open_soma(census_version="latest", context=small_mem_context) as census:
        with census["census_data"][experiment_name].axis_query(
            measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
        ) as query:
            mean_variance = pp.mean_variance(query, calculate_mean=False, calculate_variance=True)
            assert isinstance(mean_variance, pd.DataFrame)
            assert "mean" not in mean_variance
            assert "variance" in mean_variance
            assert mean_variance.index.name == "soma_joinid"

            # Pick one element from the dataframe
            test_soma_joinid = int(mean_variance.index[0])  # type: ignore

            with census["census_data"][experiment_name].axis_query(
                measurement_name="RNA", obs_query=soma.AxisQuery(coords=([test_soma_joinid],))
            ) as row_query:
                sparse_row = row_query.X("raw").coos((1 + test_soma_joinid, row_query.n_vars)).concat()
                sparse_mat = sparse_row.to_scipy()
                row = sparse_mat.tocsr().getrow(test_soma_joinid)

                assert math.isclose(
                    row.todense().var(), mean_variance.loc[test_soma_joinid, "variance"], rel_tol=0.01  # type: ignore
                )
