import numpy as np
from scipy.sparse import csr_matrix

from tools.cell_census_builder.util import is_positive_integral


def test_is_positive_integral() -> None:
    X = np.array([1, 2, 3, 4])
    assert is_positive_integral(X)

    X = np.array([-1, 2, 3, 4])
    assert not is_positive_integral(X)

    X = np.array([1.2, 0, 3, 4])
    assert not is_positive_integral(X)

    X = np.zeros((3, 4))
    assert is_positive_integral(X)

    X = csr_matrix([[1, 2, 3], [4, 5, 6]])
    assert is_positive_integral(X)

    X = csr_matrix([[-1, 2, 3], [4, 5, 6]])
    assert not is_positive_integral(X)

    X = csr_matrix([[1.2, 0, 3], [4, 5, 6]])
    assert not is_positive_integral(X)

    X = csr_matrix([0, 0, 0])
    assert is_positive_integral(X)
