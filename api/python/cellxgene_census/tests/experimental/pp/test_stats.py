from typing import Any, Tuple, Union

import numpy as np
import numpy.ma as ma
import pandas as pd
import pytest
import tiledbsoma as soma
from scipy import sparse

import cellxgene_census
from cellxgene_census.experimental import pp


def var(X: Union[sparse.csc_matrix, sparse.csr_matrix], axis: int = 0, ddof: int = 1) -> Any:
    """
    Variance of a sparse matrix calculated as mean(X**2) - mean(X)**2
    with Bessel's correction applied for unbiased estimate
    """
    X_squared = X.copy()
    X_squared.data **= 2
    n = X.shape[axis]
    return ((X_squared.sum(axis=axis).A1 / n) - np.square(X.sum(axis=axis).A1 / n)) * (n / (n - ddof))


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("calc_mean,calc_variance", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize(
    "experiment_name,obs_value_filter,obs_coords",
    [
        ("mus_musculus", 'tissue_general == "liver" and is_primary_data == True', ()),
        ("mus_musculus", 'is_primary_data == True and tissue_general == "heart"', ()),
        pytest.param(
            "mus_musculus",
            "is_primary_data == True",
            (slice(0, 400_000),),
            marks=pytest.mark.expensive,
        ),
        pytest.param(
            "homo_sapiens",
            "is_primary_data == True",
            (slice(0, 400_000),),
            marks=pytest.mark.expensive,
        ),
    ],
)
def test_mean_variance(
    experiment_name: str,
    obs_value_filter: str,
    axis: int,
    calc_mean: bool,
    calc_variance: bool,
    small_mem_context: soma.SOMATileDBContext,
    obs_coords: Tuple[None, slice],
) -> None:
    with cellxgene_census.open_soma(census_version="latest", context=small_mem_context) as census:
        with census["census_data"][experiment_name].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=obs_value_filter, coords=obs_coords),
        ) as query:
            mean_variance = pp.mean_variance(
                query,
                calculate_mean=calc_mean,
                calculate_variance=calc_variance,
                axis=axis,
            )
            assert isinstance(mean_variance, pd.DataFrame)
            if calc_mean:
                assert "mean" in mean_variance
                assert mean_variance["mean"].dtype == np.float64

            if calc_variance:
                assert "variance" in mean_variance
                assert mean_variance["variance"].dtype == np.float64

            if not calc_mean:
                assert "mean" not in mean_variance
            if not calc_variance:
                assert "variance" not in mean_variance

            assert mean_variance.index.name == "soma_joinid"
            if axis == 0:
                assert np.array_equal(mean_variance.index, query.var_joinids())
            else:
                assert np.array_equal(mean_variance.index, query.obs_joinids())

            table = query.X("raw").tables().concat()
            data = table["soma_data"].to_numpy()

            dim_0 = query.indexer.by_obs(table["soma_dim_0"])
            dim_1 = query.indexer.by_var(table["soma_dim_1"])
            coo = sparse.coo_matrix((data, (dim_0, dim_1)), shape=(query.n_obs, query.n_vars))

            if calc_mean:
                mean = coo.mean(axis=axis)
                if axis == 1:
                    mean = mean.T
                assert np.allclose(mean, mean_variance["mean"], atol=1e-5, rtol=1e-2)

            if calc_variance:
                variance = var(coo, axis=axis)
                assert np.allclose(variance, mean_variance["variance"], atol=1e-5, rtol=1e-2)


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("calc_mean,calc_variance", [(True, True)])
@pytest.mark.parametrize(
    "experiment_name,obs_coords",
    [
        ("mus_musculus", (slice(0, 1_000),)),
    ],
)
def test_mean_variance_nnz_only(
    experiment_name: str,
    axis: int,
    calc_mean: bool,
    calc_variance: bool,
    small_mem_context: soma.SOMATileDBContext,
    obs_coords: Tuple[None, slice],
) -> None:
    # Note: since this test requires materializing the matrix in memory to compute the mean/variance,
    # we're going to use a coord slice based approach. This will ensure the matrix can fit in memory.
    with cellxgene_census.open_soma(census_version="latest", context=small_mem_context) as census:
        with census["census_data"][experiment_name].axis_query(
            measurement_name="RNA", obs_query=soma.AxisQuery(coords=obs_coords)
        ) as query:
            mean_variance = pp.mean_variance(
                query,
                calculate_mean=calc_mean,
                calculate_variance=calc_variance,
                axis=axis,
                nnz_only=True,
                ddof=0,
            )

            table = query.X("raw").tables().concat()
            data = table["soma_data"].to_numpy()

            dim_0 = query.indexer.by_obs(table["soma_dim_0"])
            dim_1 = query.indexer.by_var(table["soma_dim_1"])
            coo = sparse.coo_matrix((data, (dim_0, dim_1)), shape=(query.n_obs, query.n_vars))

            dense = coo.toarray()

            mask = np.ones(coo.shape)
            r, c = coo.nonzero()
            for x, y in zip(r, c):
                mask[x, y] = 0
            masked = ma.masked_array(dense, mask=mask)  # type: ignore[no-untyped-call, var-annotated]

            if calc_mean:
                mean = masked.mean(axis=axis)  # type: ignore[no-untyped-call]
                assert np.allclose(mean, mean_variance["mean"], atol=1e-5, rtol=1e-1, equal_nan=True)

            if calc_variance:
                variance = masked.var(axis=axis, ddof=0)  # type: ignore[no-untyped-call]
                va = mean_variance["variance"].to_numpy()
                assert np.allclose(variance, va, atol=1e-5, rtol=1e-2, equal_nan=True)


@pytest.mark.experimental
def test_mean_variance_no_flags() -> None:
    with pytest.raises(ValueError):
        pp.mean_variance(soma.AxisQuery(), calculate_mean=False, calculate_variance=False)


@pytest.mark.parametrize("experiment_name", ["mus_musculus"])
def test_mean_variance_empty_query(experiment_name: str, small_mem_context: soma.SOMATileDBContext) -> None:
    with cellxgene_census.open_soma(census_version="latest", context=small_mem_context) as census:
        with census["census_data"][experiment_name].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter='tissue_general == "foo"'),
        ) as query:
            with pytest.raises(ValueError):
                pp.mean_variance(query, calculate_mean=True, calculate_variance=True)


@pytest.mark.experimental
def test_mean_variance_wrong_axis() -> None:
    with pytest.raises(ValueError):
        pp.mean_variance(soma.AxisQuery(), calculate_mean=True, calculate_variance=True, axis=2)
