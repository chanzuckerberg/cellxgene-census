import numpy as np
import numpy.typing as npt
import pytest
from scipy import sparse

from cellxgene_census.experimental.pp._online import MeanVarianceAccumulator


def allclose(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> bool:
    return np.allclose(a, b, atol=1e-5, rtol=1e-3, equal_nan=True)


@pytest.fixture
def matrix(m: int, n: int) -> sparse.coo_matrix:
    m = 100 * sparse.random(m, n, density=0.1, format="coo", dtype=np.float32, random_state=np.random.default_rng())
    m.row.flags.writeable = False  # type: ignore[attr-defined]
    m.col.flags.writeable = False  # type: ignore[attr-defined]
    m.data.flags.writeable = False  # type: ignore[attr-defined]
    return m


@pytest.mark.experimental
@pytest.mark.parametrize("stride", [101, 53])
@pytest.mark.parametrize("n_batches", [1, 3, 11, 101])
@pytest.mark.parametrize("m,n", [(1200, 511), (100001, 57)])
def test_meanvar(matrix: sparse.coo_matrix, n_batches: int, stride: int) -> None:
    rng = np.random.default_rng()
    batches_prob = rng.random(n_batches)
    batches_prob /= batches_prob.sum()
    batches = rng.choice(n_batches, matrix.shape[0], p=batches_prob)
    batches.flags.writeable = False

    batch_id, batch_count = np.unique(batches, return_counts=True)
    n_samples = np.zeros((n_batches,), dtype=np.int64)
    n_samples[batch_id] = batch_count
    assert n_samples.sum() == matrix.shape[0]
    assert len(n_samples) == n_batches

    olmv = MeanVarianceAccumulator(n_batches, n_samples, matrix.shape[1])
    for i in range(0, matrix.nnz, stride):
        olmv.update_by_batch(
            batches[matrix.row[i : i + stride]], matrix.col[i : i + stride], matrix.data[i : i + stride]
        )
    batches_u, batches_var, all_u, all_var = olmv.finalize()

    assert isinstance(all_u, np.ndarray)
    assert isinstance(all_var, np.ndarray)
    assert isinstance(batches_u, np.ndarray)
    assert isinstance(batches_var, np.ndarray)

    assert all_u.shape == (matrix.shape[1],)
    assert all_var.shape == (matrix.shape[1],)
    assert batches_u.shape == (n_batches, matrix.shape[1])
    assert batches_var.shape == (n_batches, matrix.shape[1])

    dense = matrix.toarray()
    assert allclose(all_u, dense.mean(axis=0))
    assert allclose(all_var, dense.var(axis=0, ddof=1, dtype=np.float64))

    matrix = matrix.tocsr()  # COO not conveniently indexable
    for batch in range(n_batches):
        dense = matrix[batches == batch, :].toarray()
        assert allclose(batches_u[batch], dense.mean(axis=0))
        assert allclose(batches_var[batch], dense.var(axis=0, ddof=1, dtype=np.float64))
