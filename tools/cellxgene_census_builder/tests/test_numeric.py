import numpy as np
import pytest

from cellxgene_census_builder.build_soma.experiment_builder import _roundHalfToEven


@pytest.mark.parametrize("keepbits", list(range(8, 23)))
def test_roundHalfToEven_CheckClearBits(keepbits: int) -> None:
    """Verify the bits expected to be zero are in fact zero, and that the rounding did round toward even."""
    assert keepbits > 0 and keepbits < 23
    A = np.arange(-9999, 10000, dtype=np.float32) / 9999.999
    rA = _roundHalfToEven(A.copy(), keepbits=keepbits)

    mantissa_bits = 23
    expect_clear_mask = (1 << (mantissa_bits - keepbits)) - 1
    assert ((rA.view(dtype=np.uint32) & expect_clear_mask) == 0).all()

    assert len(A) % 2 == 1
    assert (A - rA).mean() == 0.0


def test_roundHalfToEvenDTypeCheck() -> None:
    with pytest.raises(AssertionError):
        _roundHalfToEven(np.arange(0, 10, dtype=np.float64), keepbits=15)


def test_roundHalfToEvenOutOfBoundsKeepbits() -> None:
    A = np.arange(0, 100, dtype=np.float32) / 7.0
    np.array_equal(A, _roundHalfToEven(A.copy(), keepbits=0))
    np.array_equal(A, _roundHalfToEven(A.copy(), keepbits=24))
