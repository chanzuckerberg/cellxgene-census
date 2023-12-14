"""
Validate an embedding
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Generator, Tuple, TypeVar, Union, cast

import numba as nb
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import tiledbsoma as soma

from .census_util import get_census_obs_uri_region, get_obs_soma_joinids
from .config import Config
from .util import EagerIterator, blocksize, blockwise_axis0_tables, get_logger, has_blockwise_iterator, soma_context

_NPT = TypeVar("_NPT", bound=npt.NDArray[np.number[Any]])


"""
Multiple implementations of embedding validation. All require metadata as well to ensure
it matches the embedding. Tests performed:

1. Embedding shape must be (O, M), where O is the domain of the associated Census experiment
obs dataframe, and M is user defined (e.g., for a 2D UMAP, M would be 2).
2. All dim0 values in the embedding must be a legal obs soma_joinid in the corresponding Census experiment
3. An embedding must have at least one (1) cell embedded
4. Embedding data type must be float32 and coordinates must be int64
5. Storage format version must match associated Census

"""

logger = get_logger()


def validate_compatible_tiledb_storage_format(uri: str, config: Config) -> None:
    """Verify Census build and Embedding TileDB formats are identical"""
    import tiledb

    # Fetch embedding storage version
    emb_storage_version = tiledb.open(uri).schema.version

    # Fetch associated Census array URI and its associated storage version
    census_obs_uri, census_region = get_census_obs_uri_region(config)
    census_storage_version = tiledb.open(census_obs_uri, config={"vfs.s3.region": census_region}).schema.version

    if emb_storage_version != census_storage_version:
        raise ValueError("tiledb storage versions for embedding and Census are mismatched")


def validate_embedding(config: Config, uri: str) -> None:
    """
    Validate an embedding saved as a SOMASparseNDArray, e.g., obsm-like. Raises on invalid
    """
    logger.info(f"Validating {uri}")
    metadata = config.metadata
    obs_joinids, _ = get_obs_soma_joinids(config)

    with soma.open(uri, context=soma_context()) as A:
        shape = A.shape

        # Verify type: float32
        if A.schema.field("soma_data").type != pa.float32():
            raise ValueError("Embedding data type not float32")

        _validate_shape(shape, config)

        # Must have at least one cell embedded
        if A.nnz == 0:
            raise ValueError("Embedding must have at least one cell embedded")

        # coroutine that keeps track of sorted dup state
        check_dups = is_sorted_unique_gen(shape[1])
        next(check_dups)  # prime the generator

        size = blocksize(shape[1])
        counted_embeddings = 0
        with concurrent.futures.ThreadPoolExecutor() as tp:
            for tbl, _ in EagerIterator(
                A.read(result_order="row-major").blockwise(axis=0, size=size, reindex_disable_on_axis=[0, 1]).tables()
                if has_blockwise_iterator()
                else blockwise_axis0_tables(A, result_order="row-major", size=size, reindex_disable_on_axis=[0, 1]),
                pool=tp,
            ):
                logger.debug(f"Read table, length {len(tbl)}")
                i = tbl.column("soma_dim_0")
                j = tbl.column("soma_dim_1")

                _in_all = tp.submit(isin_all, i, obs_joinids)
                _in_range = tp.submit(is_in_range_all, j, 0, metadata.n_features - 1)

                # Keep count of dim0 coordinates seen
                counted_embeddings += len(np.unique(i.to_numpy()))

                # Embedding must contain no dups
                no_dups = check_dups.send(tbl)
                if not no_dups:
                    raise ValueError("Embedding must not contain duplicate coordinates")

                # verify all dim0 values are legit obs soma_joinid values
                if not _in_all.result():
                    raise ValueError("Embedding contains joinids not present in experiment obs")

                # Verify all dim1 values are in expected domain
                if not _in_range.result():
                    raise ValueError("Embedding dim_1 values not in range [0, n_features)")

        if counted_embeddings != metadata.n_embeddings:
            raise ValueError(
                f"Number of embeddings in data [{counted_embeddings}] and metadata [{metadata.n_embeddings}] do not match."
            )


def _validate_shape(shape: Tuple[int, ...], config: Config) -> None:
    _, obs_shape = get_obs_soma_joinids(config)

    if len(shape) != 2:
        raise ValueError("Embedding must be 2D")

    if shape[0] != obs_shape[0]:
        raise ValueError(f"Shapes differ: embedding {shape[0]},  obs shape {obs_shape[0]}.")

    if shape[1] != config.metadata.n_features:
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
