from __future__ import annotations

import functools
import logging
import math
import pathlib
import urllib
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.metadata import metadata
from typing import Any, Dict, Generator, Iterator, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp
import tiledbsoma as soma
from packaging.version import Version


@functools.cache
def has_blockwise_iterator() -> bool:
    """
    Feature flag. Return true if the tiledbsoma SparseNDArray contains the blockwise iterator.
    Introduced in version 1.5.
    """
    return cast(bool, Version(metadata("tiledbsoma")["Version"]) >= Version("1.5.0"))


def get_logger() -> logging.Logger:
    return logging.getLogger("census_contrib")


DEFAULT_READ_BUFFER_SIZE = 4 * 1024**3
MAX_NNZ_GOAL = DEFAULT_READ_BUFFER_SIZE // 8  # sizeof(int64) - worst case size


def blocksize(n_features: int, nnz_goal: int = MAX_NNZ_GOAL) -> int:
    """
    Given an nnz goal, and n_features, return step size for a blockwise iterator.
    """
    nnz_goal = max(nnz_goal, MAX_NNZ_GOAL)
    return cast(int, 2 ** round(math.log2((nnz_goal) / n_features)))


def soma_context(tiledb_config: Optional[Dict[str, Any]] = None) -> soma.options.SOMATileDBContext:
    """Return soma context with default config."""
    tiledb_config = tiledb_config or {}
    return soma.options.SOMATileDBContext().replace(
        tiledb_config={
            "py.init_buffer_bytes": DEFAULT_READ_BUFFER_SIZE + 10 * 1024,
            "soma.init_buffer_bytes": DEFAULT_READ_BUFFER_SIZE + 10 * 1024,
            **tiledb_config,
        }
    )


def uri_to_path(uri: str) -> pathlib.Path:
    assert uri.startswith("file://")
    return pathlib.Path(urllib.parse.unquote(urllib.parse.urlparse(uri).path))


_T = TypeVar("_T")


class EagerIterator(Iterator[_T]):
    def __init__(
        self,
        iterator: Iterator[_T],
        pool: Optional[ThreadPoolExecutor] = None,
    ):
        super().__init__()
        self.iterator = iterator
        self._pool = pool or ThreadPoolExecutor()
        self._own_pool = pool is None
        self._future: Optional[Future[_T]] = None
        self._fetch_next()

    def _fetch_next(self) -> None:
        self._future = self._pool.submit(self.iterator.__next__)

    def __next__(self) -> _T:
        try:
            assert self._future
            res = self._future.result()
            self._fetch_next()
            return res
        except StopIteration:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        if self._own_pool:
            self._pool.shutdown()

    def __del__(self) -> None:
        # Ensure the threadpool is cleaned up in the case where the
        # iterator is not exhausted. For more information on __del__:
        # https://docs.python.org/3/reference/datamodel.html#object.__del__
        self._cleanup()
        super_del = getattr(super(), "__del__", lambda: None)
        super_del()


#
# ----------
# backport from tiledbsoma 1.5 as a temporary means of running on prior versions.
#
# Remove once the Census builder is on tiledbsoma>=1.5.0
# ----------
#
def blockwise_axis0_tables(
    A: soma.SparseNDArray,
    coords: soma.options.SparseNDCoords = (),
    result_order: soma.options.ResultOrderStr = soma.ResultOrder.AUTO,
    size: Optional[Union[int, Sequence[int]]] = None,
    reindex_disable_on_axis: Optional[Union[int, Sequence[int]]] = None,
) -> Generator[Tuple[pa.Table, Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]], None, None]:
    assert A.ndim == 2
    coords, size, reindex_disable_on_axis = _validate_args(A.shape, coords, size, reindex_disable_on_axis)
    minor_joinids = pa.array(np.concatenate(list(_coords_strider(coords[1], A.shape[1], A.shape[1]))))

    coords_reader = (coord_chunk for coord_chunk in _coords_strider(coords[0], A.shape[0], size[0]))
    table_reader = EagerIterator(
        (A.read(coords=(coord_chunk,), result_order=result_order).tables().concat(), coord_chunk)
        for coord_chunk in coords_reader
    )

    for tbl, coord_chunk in table_reader:
        # ... reindexing
        pytbl = {}
        for dim in range(2):
            col = tbl.column(f"soma_dim_{dim}").to_numpy()
            if dim not in reindex_disable_on_axis:
                col = pd.Index(coord_chunk).get_indexer(col)  # type: ignore
            pytbl[f"soma_dim_{dim}"] = col
        pytbl["soma_data"] = tbl.column("soma_data").to_numpy()
        tbl = pa.Table.from_pydict(pytbl)

        yield tbl, (coord_chunk, minor_joinids)


def blockwise_axis0_scipy_csr(
    A: soma.SparseNDArray,
    coords: soma.options.SparseNDCoords = (),
    result_order: soma.options.ResultOrderStr = soma.ResultOrder.AUTO,
    size: Optional[Union[int, Sequence[int]]] = None,
    reindex_disable_on_axis: Optional[Union[int, Sequence[int]]] = None,
) -> Generator[Tuple[sp.csr_matrix, Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]], None, None]:
    assert A.ndim == 2
    coords, size, reindex_disable_on_axis = _validate_args(A.shape, coords, size, reindex_disable_on_axis)

    for tbl, (major_coords, minor_coords) in EagerIterator(
        blockwise_axis0_tables(A, coords, result_order, size, reindex_disable_on_axis)
    ):
        tbl = tbl.sort_by([("soma_dim_0", "ascending"), ("soma_dim_1", "ascending")])
        i = tbl.column("soma_dim_0").to_numpy()
        j = tbl.column("soma_dim_1").to_numpy()
        d = tbl.column("soma_data").to_numpy()

        shape = list(A.shape)
        ij = (i, j)
        for dim in range(2):
            if dim not in reindex_disable_on_axis:
                shape[dim] = len((major_coords, minor_coords)[dim])

        yield sp.csr_matrix((d, ij), shape=shape), (major_coords, minor_coords)


_ElemT = TypeVar("_ElemT")


def _pad_with_none(s: Sequence[_ElemT], to_length: int) -> Tuple[Optional[_ElemT], ...]:
    """Given a sequence, pad length to a user-specified length, with None values"""
    return tuple(s[i] if i < len(s) else None for i in range(to_length))


def _validate_args(
    shape: Tuple[int, ...],
    coords: soma.options.SparseNDCoords,
    size: Optional[Union[int, Sequence[int]]] = None,
    reindex_disable_on_axis: Optional[Union[int, Sequence[int]]] = None,
) -> Tuple[Tuple[Any, Any], Sequence[int], Sequence[int]]:
    ndim = len(shape)
    axis = [0]

    coords = _pad_with_none(coords, ndim)

    if reindex_disable_on_axis is None:
        reindex_disable_on_axis = []
    elif isinstance(reindex_disable_on_axis, int):
        reindex_disable_on_axis = [reindex_disable_on_axis]
    elif isinstance(reindex_disable_on_axis, Sequence):
        reindex_disable_on_axis = list(reindex_disable_on_axis)
    else:
        raise TypeError("reindex_disable_on_axis must be None, int or Sequence[int]")

    default_block_size = (2**16,) + (2**8,) * (ndim - 1)
    if size is None:
        size = [default_block_size[d] for d in axis]
    elif isinstance(size, int):
        size = [size] * len(axis)
    elif isinstance(size, Sequence):
        size = list(size) + [default_block_size[d] for d in axis[len(size) :]]
    else:
        raise TypeError("blockwise iterator `size` must be None, int or Sequence[int]")

    return coords, size, reindex_disable_on_axis


def _coords_strider(coords: soma.options.SparseNDCoord, length: int, stride: int) -> Iterator[npt.NDArray[np.int64]]:
    """
    Private.

    Iterate over major coordinates, in stride sized steps, materializing each step as an
    ndarray of coordinate values. Will be sorted in ascending order.

    NB: SOMA slices are _closed_ (i.e., inclusive of both range start and stop)
    """

    # normalize coord to either a slice or ndarray

    # NB: type check on slice is to handle the case where coords is an NDArray,
    # and the equality check is broadcast to all elements of the array.
    if coords is None or (isinstance(coords, slice) and coords == slice(None)):
        coords = slice(0, length - 1)
    elif isinstance(coords, int):
        coords = np.array([coords], dtype=np.int64)
    elif isinstance(coords, Sequence):
        coords = np.array(coords).astype(np.int64)
    elif isinstance(coords, (pa.Array, pa.ChunkedArray)):
        coords = coords.to_numpy()
    elif not isinstance(coords, (np.ndarray, slice)):
        raise TypeError("Unsupported slice coordinate type")

    if isinstance(coords, slice):
        soma._util.validate_slice(coords)  # NB: this enforces step == 1, assumed below
        start, stop, _step = coords.indices(length - 1)
        assert _step == 1
        yield from (np.arange(i, min(i + stride, stop + 1), dtype=np.int64) for i in range(start, stop + 1, stride))

    else:
        assert isinstance(coords, np.ndarray) and coords.dtype == np.int64
        for i in range(0, len(coords), stride):
            yield cast(npt.NDArray[np.int64], coords[i : i + stride])


#
# ----------
# end of blockwise iter backport
# ----------
#
