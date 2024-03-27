import numpy as np
import numpy.ma as ma
import pytest
import scipy.sparse as sparse

from cellxgene_census_builder.build_soma.stats import get_obs_stats, get_var_stats


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_get_obs_stats(format: str) -> None:
    X = sparse.random(1000, 100, format=format, dtype=np.float32)
    df = get_obs_stats(X)

    assert (df.raw_sum == X.sum(axis=1, dtype=np.float64).A1).all()
    assert (df.nnz == X.getnnz(axis=1)).all()

    raw_mean_nnz = X.sum(axis=1, dtype=np.float64).A1 / X.getnnz(axis=1)
    raw_mean_nnz[np.isnan(raw_mean_nnz)] = 0.0
    assert (df.raw_mean_nnz == raw_mean_nnz).all()

    Xarr = X.toarray()
    Xmasked = ma.masked_array(Xarr, Xarr == 0)
    raw_variance_nnz = Xmasked.var(axis=1, ddof=1, dtype=np.float64).filled(0.0)
    assert np.allclose(df.raw_variance_nnz, raw_variance_nnz)

    assert (df.n_measured_vars == -1).all()


@pytest.mark.parametrize("format", ["csr", "csc", "np"])
def test_get_var_stats(format: str) -> None:
    if format == "np":
        X = np.random.default_rng().random((1000, 100), dtype=np.float32)
        nnz = np.count_nonzero(X, axis=0)
    else:
        X = sparse.random(1000, 100, format=format, dtype=np.float32)
        nnz = X.getnnz(axis=0)

    df = get_var_stats(X)

    assert (df.nnz == nnz).all()
    assert (df.n_measured_obs == 0).all()


def test_expected_errors() -> None:
    with pytest.raises(NotImplementedError):
        get_obs_stats(np.ones((10, 2), dtype=np.float32))

    with pytest.raises(NotImplementedError):
        get_var_stats([0, 1, 2])
