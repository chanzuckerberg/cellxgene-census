"""
Validate an embedding
"""
from __future__ import annotations

from typing import Any, Generator, Tuple, TypeVar, Union, cast

import numba as nb
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import tiledbsoma as soma

from .census_util import get_obs_soma_joinids
from .embedding import EmbeddingIJD
from .metadata import EmbeddingMetadata
from .util import EagerIterator, get_logger, soma_context

_NPT = TypeVar("_NPT", bound=npt.NDArray[np.number[Any]])


"""
Multiple implementations of embedding validation. All require metadata as well to ensure
it matches the embedding. Tests performed:

1. Embedding shape must be (O, M), where O is the domain of the associated Census experiment
obs dataframe, and M is user defined (e.g., for a 2D UMAP, M would be 2).
2. All dim0 values in the embedding must be a legal obs soma_joinid in the corresponding Census experiment
3. An embedding must have at least one (1) cell embedded
4. Embedding data type must be float32 and coordinates must be int64

"""

logger = get_logger()


def validate_embedding(uri: str, metadata: EmbeddingMetadata) -> None:
    """
    Validate an embedding saved as a SOMASparseNDArray, e.g., obsm-like. Raises on invalid
    """
    obs_joinids, _ = get_obs_soma_joinids(metadata)

    with soma.open(uri, context=soma_context()) as A:
        shape = A.shape

        # Verify type: float32
        if A.schema.field("soma_data").type != pa.float32():
            raise ValueError("Embedding data type not float32")

        _validate_shape(shape, metadata)

        # Must have at least one cell embedded
        if A.nnz == 0:
            raise ValueError("Embedding must have at least one cell embedded")

        # coroutine that keeps track of sorted dup state
        check_dups = is_sorted_unique_gen(shape[1])
        next(check_dups)  # prime the generator

        for tbl, _ in EagerIterator(
            A.read(result_order="row-major").blockwise(axis=0, size=2**20, reindex_disable_on_axis=[0, 1]).tables()
        ):
            logger.debug(f"Read table, length {len(tbl)}")
            i = tbl.column("soma_dim_0")
            j = tbl.column("soma_dim_1")

            # verify all dim0 values are legit obs soma_joinid values
            if not isin_all(i, obs_joinids):
                raise ValueError("Embedding contains joinids not present in experiment obs")

            # Verify all dim1 values are in expected domain
            if not is_in_range_all(j, 0, metadata.n_features - 1):
                raise ValueError("Embedding dim_1 values not in range [0, n_features)")

            # Embedding must contain no dups
            no_dups = check_dups.send(tbl)
            if not no_dups:
                raise ValueError("Embedding must not contain duplicate coordinates")


def validate_embeddingIJD(emb: EmbeddingIJD, metadata: EmbeddingMetadata) -> None:
    """
    Validate an in-memory EmbeddingIJD. Raises on invalid.
    """

    obs_joinids, _ = get_obs_soma_joinids(metadata)

    # Verify types
    if emb.i.dtype != np.int64 or emb.j.dtype != np.int64 or emb.d.dtype != np.float32:
        raise ValueError("Embedding data types not int64/int64/float32")

    # Embedding shape
    _validate_shape(emb.shape, metadata)
    if emb.i.shape != emb.j.shape or emb.i.shape != emb.d.shape:
        raise ValueError("Malformed embedding COO")

    # Must have at least one cell embedded
    if len(emb.i) < 1:
        raise ValueError("Embedding must have at least one cell embedded")

    # Verify I/dim0 values all exist as legit soma_joinids
    if not isin_all(emb.i, obs_joinids):
        raise ValueError("Embedding contains joinids not present in experiment obs")

    # Verify all J/dim1 values are in expected domain
    if not is_in_range_all(emb.j, 0, metadata.n_features - 1):
        raise ValueError("Embedding J values not in range [0, n_features)")

    # Embedding must be sorted with no dups (test assumes feature indices [0, N))
    if not is_sorted_unique(emb.i, emb.j, emb.shape[1]):
        raise ValueError("Embedding must not contain duplicate coordinates")


def _validate_shape(shape: Tuple[int, ...], metadata: EmbeddingMetadata) -> None:
    _, obs_shape = get_obs_soma_joinids(metadata)

    if len(shape) != 2:
        raise ValueError("Embedding must be 2D")

    if shape[0] != obs_shape[0]:
        raise ValueError("Embedding and obs shape differ.")

    if shape[1] != metadata.n_features:
        raise ValueError(
            "Embedding and metadata specify a different number of embedding features.",
        )


@nb.njit()  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _isin_all(elmts: _NPT, test_elmts: _NPT) -> bool:
    """
    Return equivalent of numpy.isin(elmts, test_elmts).all() without the
    memory allocation and extra reduction required by the numpy expression.
    """
    test = set(test_elmts)
    for i in range(len(elmts)):
        if elmts[i] not in test:
            return False
    return True


def isin_all(elmts: Union[pa.ChunkedArray, pa.Array, _NPT], test_elmts: _NPT) -> bool:
    if isinstance(elmts, pa.ChunkedArray):
        return all(_isin_all(chunk.to_numpy(), test_elmts) for chunk in elmts.iterchunks())
    elif isinstance(elmts, pa.Array):
        return cast(bool, _isin_all(elmts.to_numpy(), test_elmts))
    else:
        return cast(bool, _isin_all(elmts, test_elmts))


@nb.njit()  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _is_in_range_all(elmts: _NPT, min: float, max: float) -> bool:
    """
    Return equivalent of np.logical_or((elmts < min), (elmts > max)).any()
    without the memory allocation and extra reduction required by the numpy expression.
    """
    for i in range(len(elmts)):
        if elmts[i] < min or elmts[i] > max:
            return False
    return True


def is_in_range_all(elmts: Union[pa.ChunkedArray, pa.Array, _NPT], min: float, max: float) -> bool:
    if isinstance(elmts, pa.ChunkedArray):
        return all(_is_in_range_all(chunk.to_numpy(), min, max) for chunk in elmts.iterchunks())
    elif isinstance(elmts, pa.Array):
        return cast(bool, _is_in_range_all(elmts.to_numpy(), min, max))
    else:
        return cast(bool, _is_in_range_all(elmts, min, max))


@nb.njit()  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _is_sorted_unique(
    i: npt.NDArray[np.int64], j: npt.NDArray[np.int64], j_shape: int, last_coord: int
) -> Tuple[bool, int]:
    for n in range(len(i)):
        c_coord = i[n] * j_shape + j[n]
        if c_coord <= last_coord:
            return False, last_coord
        last_coord = c_coord
    return True, last_coord


def is_sorted_unique(i: npt.NDArray[np.int64], j: npt.NDArray[np.int64], j_shape: int) -> bool:
    ok, _ = cast(Tuple[bool, int], _is_sorted_unique(i, j, j_shape, -1))
    return ok


def is_sorted_unique_gen(j_shape: int) -> Generator[bool, pa.Table, None]:
    last_coord = -1
    tbl = None
    ok = True
    while True:
        tbl = yield ok
        if tbl is None:
            break
        ok, last_coord = _is_sorted_unique(
            tbl.column("soma_dim_0").to_numpy(),
            tbl.column("soma_dim_1").to_numpy(),
            j_shape,
            last_coord,
        )
