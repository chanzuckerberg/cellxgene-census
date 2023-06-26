from typing import Generator, Iterator, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sparse
import tiledbsoma as soma
from typing_extensions import Literal

from ._eager_iter import EagerIterator

_RT = Tuple[Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], sparse.spmatrix]


def X_sparse_iter(
    query: soma.ExperimentAxisQuery,
    X_name: str = "raw",
    row_stride: int = 2**16,  # row stride
    fmt: Literal["csr", "csc"] = "csr",  # the resulting sparse format
    be_eager: bool = True,
) -> Iterator[_RT]:
    """
    Return a row-wise iterator over the user-specified X SparseNdMatrix, returning for each
    iteration step:
        * obs coords (coordinates)
        * var_coords (coordinates)
        * X contents as a SciPy csr_matrix or csc_matrix
    The coordinates and X matrix chunks are indexed positionally, i.e. for any
    given value in the matrix, X[i, j], the original soma_joinid (aka soma_dim_0
    and soma_dim_1) are present in obs_coords[i] and var_coords[j].

    Args:
        query:
            A SOMA ExperimentAxisQuery defining the coordinates over which the iterator will
            read.
        X_name:
            The name of the X layer.
        row_stride:
            The number of rows to return in each step.
        fmt:
            The SciPy sparse array layout. Supported: 'csc' and 'csr' (default).
        be_eager:
            If true, will use multiple threads to parallelize reading
            and processing. This will improve speed, but at the cost
            of some additional memory use.

    Returns:
        An iterator which iterates over a tuple of:
            obs_coords
            var_coords
            SciPy sparse matrix

    Lifecycle:
        experimental
    """
    if fmt == "csr":
        fmt_ctor = sparse.csr_matrix
    elif fmt == "csc":
        fmt_ctor = sparse.csc_matrix
    else:
        raise ValueError("fmt must be 'csr' or 'csc'")

    # Lazy partition array by chunk_size on first dimension
    obs_coords = query.obs_joinids().to_numpy()
    obs_coord_chunker = (obs_coords[i : i + row_stride] for i in range(0, len(obs_coords), row_stride))

    # Lazy read into Arrow Table. Yields (coords, Arrow.Table)
    X = query._ms.X[X_name]
    var_coords = query.var_joinids().to_numpy()
    table_reader = (
        (
            (obs_coords_chunk, var_coords),
            X.read(coords=(obs_coords_chunk, var_coords)).tables().concat(),
        )
        for obs_coords_chunk in obs_coord_chunker
    )
    if be_eager:
        table_reader = (t for t in EagerIterator(table_reader, query._threadpool))

    # lazy reindex of obs coordinates. Yields coords and (data, i, j) as numpy ndarrays
    coo_reindexer = (
        (
            (obs_coords_chunk, var_coords),
            (
                tbl["soma_data"].to_numpy(),
                pd.Index(obs_coords_chunk).get_indexer(tbl["soma_dim_0"].to_numpy()),  # type: ignore[no-untyped-call]
                query.indexer.by_var(tbl["soma_dim_1"].to_numpy()),
            ),
        )
        for (obs_coords_chunk, var_coords), tbl in table_reader
    )
    if be_eager:
        coo_reindexer = (t for t in EagerIterator(coo_reindexer, query._threadpool))

    # lazy convert Arrow table to Scipy sparse.csr_matrix
    csr_reader: Generator[_RT, None, None] = (
        (
            (obs_coords_chunk, var_coords),
            fmt_ctor(
                sparse.coo_matrix(
                    (data, (i, j)),
                    shape=(len(obs_coords_chunk), query.n_vars),
                )
            ),
        )
        for (obs_coords_chunk, var_coords), (data, i, j) in coo_reindexer
    )
    if be_eager:
        csr_reader = (t for t in EagerIterator(csr_reader, query._threadpool))

    yield from csr_reader


_T = TypeVar("_T")
