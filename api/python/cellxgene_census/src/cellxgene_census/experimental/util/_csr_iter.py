from typing import Generator, Iterator, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sparse
import tiledbsoma as soma
from typing_extensions import Literal

from ._eager_iter import _EagerIterator

_RT = Tuple[Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], sparse.spmatrix]


def X_sparse_iter(
    query: soma.ExperimentAxisQuery,
    X_name: str = "raw",
    axis: int = 0,
    stride: int = 2**16,
    fmt: Literal["csr", "csc"] = "csr",
    use_eager_fetch: bool = True,
) -> Iterator[_RT]:
    """
    Return an iterator over an X SparseNdMatrix, returning for each iteration step:
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
        axis:
            The axis to iterate over, where zero (0) is obs axis and one (1)
            is the var axis. Currently only axis 0 (obs axis) is supported.
        stride:
            The chunk size to return in each step.
        fmt:
            The SciPy sparse array layout. Supported: 'csc' and 'csr' (default).
        use_eager_fetch:
            If true, will use multiple threads to parallelize reading
            and processing. This will improve speed, but at the cost
            of some additional memory use.

    Returns:
        An iterator which iterates over a tuple of:
            (obs_coords, var_coords)
            SciPy sparse matrix

    Examples:
        >>> with cellxgene_census.open_soma() as census:
        ...     exp = census["census_data"][experiment]
        ...     with exp.axis_query(measurement_name="RNA") as query:
        ...         for (obs_soma_joinids, var_soma_joinids), X_chunk in X_sparse_iter(
        ...             query, X_name="raw", stride=1000
        ...         ):
        ...             # X_chunk is a scipy.csr_matrix of csc_matrix
        ...             # For each X_chunk[i, j], the associated soma_joinid is
        ...             # obs_soma_joinids[i] and var_soma_joinids[j]
        ...             ...

    Lifecycle:
        experimental
    """
    if fmt == "csr":
        fmt_ctor = sparse.csr_matrix
    elif fmt == "csc":
        fmt_ctor = sparse.csc_matrix
    else:
        raise ValueError("fmt must be 'csr' or 'csc'")

    if axis != 0:
        raise ValueError("axis must be zero (obs)")

    # Lazy partition array by chunk_size on first dimension
    obs_coords = query.obs_joinids().to_numpy()
    obs_coord_chunker = (obs_coords[i : i + stride] for i in range(0, len(obs_coords), stride))

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
    if use_eager_fetch:
        table_reader = (t for t in _EagerIterator(table_reader, query._threadpool))

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
    if use_eager_fetch:
        coo_reindexer = (t for t in _EagerIterator(coo_reindexer, query._threadpool))

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
    if use_eager_fetch:
        csr_reader = (t for t in _EagerIterator(csr_reader, query._threadpool))

    yield from csr_reader
