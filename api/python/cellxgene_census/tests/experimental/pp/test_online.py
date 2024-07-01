import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import pytest
from scipy import sparse

from cellxgene_census.experimental.pp._online import (
    CountsAccumulator,
    MeanAccumulator,
    MeanVarianceAccumulator,
)


def allclose(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> bool:
    return np.allclose(a, b, atol=1e-5, rtol=1e-2, equal_nan=True)


@pytest.fixture
def matrix(m: int, n: int) -> sparse.coo_matrix:
    m = 100 * sparse.random(
        m,
        n,
        density=0.1,
        format="coo",
        dtype=np.float32,
        random_state=np.random.default_rng(),
    )
    m.row.flags.writeable = False  # type: ignore[attr-defined]
    m.col.flags.writeable = False  # type: ignore[attr-defined]
    m.data.flags.writeable = False  # type: ignore[attr-defined]
    return m


@pytest.mark.experimental
@pytest.mark.parametrize("stride", [101, 53])
@pytest.mark.parametrize("n_batches", [1, 3, 11, 101])
@pytest.mark.parametrize("m,n", [(1200, 511), (100001, 57)])
@pytest.mark.parametrize("ddof", [0, 1, 100])
@pytest.mark.parametrize("nnz_only", [True, False])
def test_meanvar(matrix: sparse.coo_matrix, n_batches: int, stride: int, ddof: int, nnz_only: bool) -> None:
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

    # nnz_only only if there is a single batch
    should_nnz_only = nnz_only and n_batches == 1

    olmv = MeanVarianceAccumulator(n_batches, n_samples, matrix.shape[1], ddof, nnz_only=should_nnz_only)
    for i in range(0, matrix.nnz, stride):
        batch_vec = batches[matrix.row[i : i + stride]] if n_batches > 1 else None
        olmv.update(matrix.col[i : i + stride], matrix.data[i : i + stride], batch_vec)
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

    if should_nnz_only:
        mask = np.ones(matrix.shape)
        r, c = matrix.tolil().nonzero()
        for x, y in zip(r, c):
            mask[x, y] = 0
        masked: np.ma.MaskedArray[np.bool_, np.dtype[np.float64]] = ma.masked_array(dense, mask=mask)  # type: ignore[no-untyped-call]
        mean = masked.mean(axis=0)  # type: ignore[no-untyped-call]
        assert allclose(all_u, mean)

        nv = masked.var(axis=0, ddof=ddof)  # type: ignore[no-untyped-call]
        assert allclose(all_var, nv)

    else:
        assert allclose(all_u, dense.mean(axis=0))
        assert allclose(all_var, dense.var(axis=0, ddof=ddof, dtype=np.float64))

        matrix = matrix.tocsr()  # COO not conveniently indexable
        for batch in range(n_batches):
            dense = matrix[batches == batch, :].toarray()
            assert allclose(batches_u[batch], dense.mean(axis=0))
            assert allclose(batches_var[batch], dense.var(axis=0, ddof=ddof, dtype=np.float64))


@pytest.mark.experimental
@pytest.mark.parametrize("m,n", [(1200, 511), (100001, 57)])
def test_meanvar_nnz_only_batches_fails(matrix: sparse.coo_matrix) -> None:
    n_batches = 10
    nnz_only = True
    n_samples = np.zeros((n_batches,), dtype=np.int64)
    ddof = 1
    with pytest.raises(ValueError):
        MeanVarianceAccumulator(n_batches, n_samples, matrix.shape[1], ddof, nnz_only=nnz_only)


@pytest.mark.experimental
@pytest.mark.parametrize("stride", [101, 53])
@pytest.mark.parametrize("m,n", [(1200, 511), (100001, 57)])
def test_mean(matrix: sparse.coo_matrix, stride: int) -> None:
    m_acc = MeanAccumulator(matrix.shape[0], matrix.shape[1])
    for i in range(0, matrix.nnz, stride):
        m_acc.update(matrix.col[i : i + stride], matrix.data[i : i + stride])
    u = m_acc.finalize()

    assert isinstance(u, np.ndarray)
    assert u.shape == (matrix.shape[1],)

    dense = matrix.toarray()
    assert allclose(u, dense.mean(axis=0))


@pytest.mark.experimental
@pytest.mark.parametrize("stride", [101, 53])
@pytest.mark.parametrize("n_batches", [1, 3, 11, 101])
@pytest.mark.parametrize("m,n", [(1200, 511), (100001, 57)])
def test_counts(matrix: sparse.coo_matrix, n_batches: int, stride: int) -> None:
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

    clip_val = 50 * np.random.rand(n_batches, matrix.shape[1])

    ca = CountsAccumulator(n_batches, matrix.shape[1], clip_val)
    for i in range(0, matrix.nnz, stride):
        batch_vec = batches[matrix.row[i : i + stride]] if n_batches > 1 else None
        ca.update(matrix.col[i : i + stride], matrix.data[i : i + stride], batch_vec)
    counts_sum, counts_squared_sum = ca.finalize()

    assert isinstance(counts_sum, np.ndarray)
    assert isinstance(counts_squared_sum, np.ndarray)

    assert counts_sum.shape == (n_batches, matrix.shape[1])
    assert counts_squared_sum.shape == (n_batches, matrix.shape[1])

    matrix = matrix.tocsr()  # COO not conveniently indexable
    for batch in range(n_batches):
        dense = matrix[batches == batch, :].toarray()
        assert allclose(counts_sum[batch], np.minimum(dense, clip_val[batch]).sum(axis=0))
        assert allclose(
            counts_squared_sum[batch],
            (np.minimum(dense, clip_val[batch]) ** 2).sum(axis=0),
        )


@pytest.mark.experimental
def test_mean_fails_no_variables_or_samples() -> None:
    with pytest.raises(ValueError, match=r"No samples provided - can't calculate mean."):
        MeanAccumulator(n_samples=0, n_variables=100)
    with pytest.raises(ValueError, match=r"No variables provided - can't calculate mean."):
        MeanAccumulator(n_samples=1000, n_variables=0)
