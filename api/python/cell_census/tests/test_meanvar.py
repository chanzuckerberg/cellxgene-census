import numpy as np
from scipy import sparse

from cell_census.compute import OnlineMatrixMeanVariance


def test_online_mean_var() -> None:

    rng = np.random.default_rng()
    matrix = sparse.random(1200, 51, density=0.1, format="coo", dtype=np.float32, random_state=rng)
    olmv = OnlineMatrixMeanVariance(*matrix.shape)

    stride = 101
    for i in range(0, matrix.nnz, stride):
        olmv.update(matrix.col[i : i + stride], matrix.data[i : i + stride])
    u, var = olmv.finalize()

    dense = matrix.toarray()
    assert np.allclose(u, dense.mean(axis=0))
    assert np.allclose(var, dense.var(axis=0), rtol=1e-3)
