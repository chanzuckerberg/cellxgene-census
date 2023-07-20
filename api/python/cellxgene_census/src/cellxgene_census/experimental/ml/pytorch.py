import logging
import os
from contextlib import contextmanager
from datetime import timedelta
from time import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import scipy
import somacore
import tiledbsoma as soma
import torch
import torchdata.datapipes.iter as pipes
from attr import define
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from somacore.query import _fast_csr
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from ..._open import _build_soma_tiledb_context
from ..util._eager_iter import _EagerIterator

pytorch_logger = logging.getLogger("cellxgene_census.experimental.pytorch")


# TODO: Rename to reflect the correct order of the Tensors within the tuple: (X, obs)
ObsAndXDatum = Tuple[Tensor, Tensor]
"""Return type of ``ExperimentDataPipe`` that pairs a Tensor of ``obs`` row(s) with a Tensor of ``X`` matrix row(s). 
The Tensors are rank 1 if ``batch_size`` is 1, otherwise the Tensors are rank 2."""


@define
class _ObsAndXSOMABatch:
    """
    Return type of ``_ObsAndXSOMAIterator`` that pairs a chunk of ``obs`` rows with the respective rows from the ``X``
    matrix.

    Lifecycle:
        experimental
    """

    obs: pd.DataFrame
    X: scipy.matrix
    stats: "Stats"

    def __len__(self) -> int:
        return len(self.obs)


Encoders = Dict[str, LabelEncoder]
"""A dictionary of ``LabelEncoder``s keyed by the ``obs`` column name."""


@define
class Stats:
    """
    Statistics about the data retrieved by ``ExperimentDataPipe`` via SOMA API. This is useful for assessing the read
    throughput of SOMA data.

    Lifecycle:
        experimental
    """

    n_obs: int = 0
    """The total number of obs rows retrieved"""

    nnz: int = 0
    """The total number of values retrieved"""

    elapsed: int = 0
    """The total elapsed time in seconds for retrieving all batches"""

    n_soma_batches: int = 0
    """The number of batches retrieved"""

    def __str__(self) -> str:
        return (
            f"n_soma_batches={self.n_soma_batches}, n_obs={self.n_obs}, nnz={self.nnz}, "
            f"elapsed={timedelta(seconds=self.elapsed)}"
        )

    def __add__(self, other: "Stats") -> "Stats":
        self.n_obs += other.n_obs
        self.nnz += other.nnz
        self.elapsed += other.elapsed
        self.n_soma_batches += other.n_soma_batches
        return self


@contextmanager
def _open_experiment(
    uri: str, aws_region: Optional[str] = None, soma_buffer_bytes: Optional[int] = None
) -> soma.Experiment:
    """Internal method for opening a SOMA ``Experiment`` as a context manager."""

    context = _build_soma_tiledb_context(aws_region)

    if soma_buffer_bytes is not None:
        context = context.replace(
            tiledb_config={
                "py.init_buffer_bytes": soma_buffer_bytes,
                "soma.init_buffer_bytes": soma_buffer_bytes,
            }
        )

    with soma.Experiment.open(uri, context=context) as exp:
        yield exp


class _ObsAndXSOMAIterator(Iterator[_ObsAndXSOMABatch]):
    obs_tables_iter: somacore.ReadIter[pa.Table]
    """Iterates the SOMA batches (tables) of corresponding ``obs`` and ``X`` data. This is an internal class, 
    not intended for public use."""

    X: soma.SparseNDArray
    """A handle to the full X data of the SOMA ``Experiment``"""

    var_joinids: pa.Array
    """The ``var`` joinids to be retrieved from the SOMA ``Experiment``"""

    def __init__(self, X: soma.SparseNDArray, obs_tables_iter: somacore.ReadIter[pa.Table], var_joinids: pa.Array):
        self.obs_tables_iter = obs_tables_iter
        self.X = X
        self.var_joinids = var_joinids

    def __next__(self) -> _ObsAndXSOMABatch:
        pytorch_logger.debug("Retrieving next SOMA batch...")
        start_time = time()

        # If no more batches to iterate through, raise StopIteration, as all iterators do when at end
        obs_table = next(self.obs_tables_iter)
        obs_batch = obs_table.to_pandas()

        # handle case of empty result (first batch has 0 rows)
        if len(obs_batch) == 0:
            raise StopIteration

        X_batch = _fast_csr.read_scipy_csr(self.X, obs_table["soma_joinid"].combine_chunks(), self.var_joinids)
        assert obs_batch.shape[0] == X_batch.shape[0]

        stats = Stats()
        stats.n_obs += X_batch.shape[0]
        stats.nnz += X_batch.nnz
        stats.elapsed += int(time() - start_time)
        stats.n_soma_batches += 1

        pytorch_logger.debug(f"Retrieved SOMA batch: {stats}")
        return _ObsAndXSOMABatch(obs=obs_batch, X=X_batch, stats=stats)


class _ObsAndXIterator(Iterator[ObsAndXDatum]):
    """
    Iterates through a set of ``obs`` and corresponding ``X`` rows, where the rows to be returned are specified by
    the ``obs_tables_iter`` argument. For the specified ``obs` rows, the corresponding ``X`` data is loaded and
    joined together. It is returned from this iterator as 2-tuples of ``X`` and obs Tensors.

    Internally manages the retrieval of data in SOMA-sized batches, fetching the next batch of SOMA data as needed.
    Supports fetching the data in an eager manner, where the next SOMA batch is fetched while the current batch is
    being read. This is an internal class, not intended for public use.
    """

    soma_batch_iter: Iterator[_ObsAndXSOMABatch]
    """The iterator for SOMA batches of paired obs and X data"""

    soma_batch: Optional[_ObsAndXSOMABatch]
    """The current SOMA batch of obs and X data"""

    i: int = -1
    """Index into current obs ``SOMA`` batch"""

    def __init__(
        self,
        X: soma.SparseNDArray,
        obs_tables_iter: somacore.ReadIter[pa.Table],
        var_joinids: npt.NDArray[np.int64],
        batch_size: int,
        encoders: Dict[str, LabelEncoder],
        stats: Stats,
        return_sparse_X: bool,
        use_eager_fetch: bool,
    ) -> None:
        self.soma_batch_iter = _ObsAndXSOMAIterator(X, obs_tables_iter, pa.array(var_joinids))
        if use_eager_fetch:
            self.soma_batch_iter = _EagerIterator(self.soma_batch_iter)
        self.soma_batch = None
        self.var_joinids = var_joinids
        self.batch_size = batch_size
        self.return_sparse_X = return_sparse_X
        self.encoders = encoders
        self.stats = stats

    def __next__(self) -> ObsAndXDatum:
        """Read the next torch batch, possibly across multiple soma batches"""

        obs: pd.DataFrame = pd.DataFrame()
        X: sparse.csr_matrix = sparse.csr_matrix((0, len(self.var_joinids)))

        while len(obs) < self.batch_size:
            try:
                obs_partial, X_partial = self._read_partial_torch_batch(self.batch_size - len(obs))
                obs = pd.concat([obs, obs_partial], axis=0)
                X = sparse.vstack([X, X_partial])
            except StopIteration:
                break

        if len(obs) == 0:
            raise StopIteration

        obs_encoded = pd.DataFrame(
            data={"soma_joinid": obs.soma_joinid}, columns=obs.columns, dtype=np.int64, index=obs.index
        )
        # TODO: Encode the entire SOMA batch at once in _read_partial_torch_batch()
        for col, enc in self.encoders.items():
            obs_encoded[col] = enc.transform(obs[col])

        # `to_numpy()` avoids copying the numpy array data
        obs_tensor = torch.from_numpy(obs_encoded.to_numpy())

        if not self.return_sparse_X:
            X_tensor = torch.from_numpy(X.todense())
        else:
            coo = X.tocoo()

            X_tensor = torch.sparse_coo_tensor(
                # Note: The `np.array` seems unnecessary, but PyTorch warns bare array is "extremely slow"
                indices=torch.from_numpy(np.array([coo.row, coo.col])),
                values=coo.data,
                size=coo.shape,
            )

        if self.batch_size == 1:
            X_tensor = X_tensor[0]
            obs_tensor = obs_tensor[0]

        return X_tensor, obs_tensor

    def _read_partial_torch_batch(self, batch_size: int) -> ObsAndXDatum:
        """Reads a torch-size batch of data from the current SOMA batch, returning a torch-size batch whose size may
        contain fewer rows than the requested ``batch_size``. This can happen when the remaining rows in the current
        SOMA batch are fewer than the requested ``batch_size``."""

        if self.soma_batch is None or not (0 <= self.i < len(self.soma_batch)):
            self.soma_batch: _ObsAndXSOMABatch = next(self.soma_batch_iter)
            self.stats += self.soma_batch.stats
            self.i = 0

            pytorch_logger.debug(f"Retrieved SOMA batch totals: {self.stats}")

        obs_batch = self.soma_batch.obs
        X_batch = self.soma_batch.X

        safe_batch_size = min(batch_size, len(obs_batch) - self.i)
        slice_ = slice(self.i, self.i + safe_batch_size)
        assert slice_.stop <= obs_batch.shape[0]

        obs_rows = obs_batch.iloc[slice_]
        assert obs_rows["soma_joinid"].is_unique
        X_csr_scipy = X_batch[slice_]
        assert safe_batch_size == obs_rows.shape[0]
        assert obs_rows.shape[0] == X_csr_scipy.shape[0]

        self.i += safe_batch_size

        return obs_rows, X_csr_scipy


class ExperimentDataPipe(pipes.IterDataPipe[Dataset[ObsAndXDatum]]):  # type: ignore
    """
    An iterable-style PyTorch ``DataPipe`` that reads ``obs`` and ``X`` data from a SOMA ``Experiment``, based upon the
    specified queries along the ``obs`` and ``var`` axes. Provides an iterator over these data when the object is
    passed to Python's built-in ``iter`` function:

    >>> for batch in iter(ExperimentDataPipe(...)):
            X_batch, y_batch = batch

    The ``batch_size`` parameter controls the number of rows of ``obs`` and ``X`` data that are returned in each
    iteration. If the ``batch_size`` is 1, then each Tensor will have rank 1:

    >>> (tensor([0., 0., 0., 0., 0., 1., 0., 0., 0.]),  # X data
         tensor([2415,    0,    0], dtype=torch.int64)) # obs data, encoded

    For larger ``batch_size`` values, the returned Tensors will have rank 2:

    >>> DataLoader(..., batch_size=3, ...):
        (tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0.],     # X batch
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
         tensor([[2415,    0,    0],                       # obs batch
                 [2416,    0,    4],
                 [2417,    0,    3]], dtype=torch.int64))

    The ``return_sparse_X`` parameter controls whether the ``X`` data is returned as a dense or sparse Tensor.  If the
    model supports use of sparse Tensors, this will reduce memory usage.

    The ``obs_column_names`` parameter determines the data columns that are returned in the ``obs`` Tensor. The first
    element is always the ``soma_joinid`` of the ``obs`` DataFrame (or, equiavalently, the ``soma_dim_0`` of the ``X``
    matrix). The remaining elements are the ``obs`` columns specified by ``obs_column_names``, and string-typed
    columns are encoded as integer values. If needed, these values can be decoded by obtaining the encoder for a
    given ``obs`` column name and calling its ``inverse_transform`` method:

    >>> exp_data_pipe.obs_encoders["<obs_attr_name>"].inverse_transform(encoded_values)

    Lifecycle:
        experimental
    """

    _initialized: bool

    _obs_joinids: Optional[npt.NDArray[np.int64]]

    _var_joinids: Optional[npt.NDArray[np.int64]]

    _encoders: Optional[Encoders]

    _stats: Stats

    # TODO: Consider adding another convenience method wrapper to construct this object whose signature is more closely
    #  aligned with get_anndata() params (i.e. "exploded" AxisQuery params).
    def __init__(
        self,
        experiment: soma.Experiment,
        measurement_name: str = "raw",
        X_name: str = "X",
        obs_query: Optional[soma.AxisQuery] = None,
        var_query: Optional[soma.AxisQuery] = None,
        obs_column_names: Sequence[str] = (),
        batch_size: int = 1,
        return_sparse_X: bool = False,
        soma_buffer_bytes: Optional[int] = None,
        use_eager_fetch: bool = True,
    ) -> None:
        """
        Construct a new ``ExperimentDataPipe``.

        Args:
            experiment:
                The SOMA ``Experiment`` from which to read data.
            measurement_name:
                The name of the SOMA ``Measurement`` to read. Defaults to "raw".
            X_name:
                The name of the X layer to read. Defaults to "X".
            obs_query:
                The query used to filter along the ``obs`` axis. If not specified, all ``obs`` and ``X`` data will
                be returned, which can be very large.
            var_query:
                The query used to filter along the ``var`` axis. If not specified, all ``var`` columns (genes/features)
                will be returned.
            obs_column_names:
                The names of the ``obs`` columns to return. The ``soma_joinid`` index "column" does not need to be
                specified and will always be returned. If not specified, only the ``soma_joinid`` will be returned.
            batch_size:
                The number of rows of ``obs`` and ``X`` data to return in each iteration. Defaults to 1. A value of 1
                will result in Tensors of rank 1 being returns (a single row); larger values will result in Tensors of
                rank 2 (multiple rows).
            return_sparse_X:
                Controls whether the ``X`` data is returned as a dense or sparse Tensor. As ``X`` data is very sparse,
                setting this to ``True`` will reduce memory usage, if the model supports use of sparse Tensors. Defaults
                to ``False``, since sparse Tensors are still experimental in PyTorch.
            soma_buffer_bytes:
                The number of bytes to use for reading data from SOMA. If not specified, will use the default SOMA
                value. Maximum memory utilization is controlled by this parameter, with larger values providing better
                read performance.
            use_eager_fetch:
                Fetch the next SOMA batch of ``obs`` and ``X`` data immediately after a previously fetched SOMA batch is made
                available for processing via the iterator. This allows network (or filesystem) requests to be made in
                parallel with client-side processing of the SOMA data, potentially improving overall performance at the
                cost of doubling memory utilization. Defaults to ``True``.
        Returns:
            The constructed ``ExperimentDataPipe``.

        Lifecycle:
            experimental
        """

        self.exp_uri = experiment.uri
        self.aws_region = experiment.context.tiledb_ctx.config().get("vfs.s3.region")
        self.measurement_name = measurement_name
        self.layer_name = X_name
        self.obs_query = obs_query
        self.var_query = var_query
        self.obs_column_names = obs_column_names
        self.batch_size = batch_size
        self.return_sparse_X = return_sparse_X
        # TODO: This will control the SOMA batch sizes, and could be replaced with a row count once TileDB-SOMA
        #  supports `batch_size` param. It affects both the obs and X read operations.
        self.soma_buffer_bytes = soma_buffer_bytes
        self.use_eager_fetch = use_eager_fetch
        self._stats = Stats()
        self._encoders = None
        self._obs_joinids = None
        self._var_joinids = None
        self._initialized = False

        if "soma_joinid" not in self.obs_column_names:
            self.obs_column_names = ["soma_joinid", *self.obs_column_names]

    def _init(self) -> None:
        if self._initialized:
            return

        pytorch_logger.debug("Initializing ExperimentDataPipe")

        with _open_experiment(self.exp_uri, self.aws_region, soma_buffer_bytes=self.soma_buffer_bytes) as exp:
            query = exp.axis_query(
                measurement_name=self.measurement_name,
                obs_query=self.obs_query,
                var_query=self.var_query,
            )

            # The to_numpy() call is a workaround for a possible bug in TileDB-SOMA:
            # https://github.com/single-cell-data/TileDB-SOMA/issues/1456
            self._obs_joinids = query.obs_joinids().to_numpy()
            self._var_joinids = query.var_joinids().to_numpy()

            self._encoders = self._build_obs_encoders(query)

        self._initialized = True

    @staticmethod
    def _partition_obs_joinids(ids: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        # NOTE: Can alternately use a `worker_init_fn` to split among workers split workload
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None or worker_info.num_workers < 2:
            return ids

        partition, num_partitions = worker_info.id, worker_info.num_workers

        partitions = np.array_split(ids, num_partitions)
        ids = partitions[partition]

        if pytorch_logger.isEnabledFor(logging.DEBUG) and len(ids) > 0:
            joinids_start = ids[0]
            joinids_end = ids[-1]
            lens = [len(p) for p in partitions]
            partition_start = sum(lens[:partition])
            partition_end_excl = partition_start + lens[partition]

            pytorch_logger.debug(
                f"Process {os.getpid()} handling partition {partition + 1} of {num_partitions}, "
                f"index range=[{partition_start}:{partition_end_excl}), "
                f"soma_joinid range=[{joinids_start}:{joinids_end}], "
                f"partition_size={len(ids)}"
            )

        return ids

    def __iter__(self) -> Iterator[ObsAndXDatum]:
        self._init()
        assert self._obs_joinids is not None
        assert self._var_joinids is not None

        if (
            self.return_sparse_X
            and torch.utils.data.get_worker_info()
            and torch.utils.data.get_worker_info().num_workers > 0
        ):
            raise NotImplementedError(
                "torch does not work with sparse tensors in multi-processing mode "
                "(see https://github.com/pytorch/pytorch/issues/20248)"
            )

        with _open_experiment(self.exp_uri, self.aws_region, soma_buffer_bytes=self.soma_buffer_bytes) as exp:
            X: soma.SparseNDArray = exp.ms[self.measurement_name].X[self.layer_name]

            obs_tables_iter = exp.obs.read(
                coords=(self._partition_obs_joinids(self._obs_joinids),),
                batch_size=somacore.BatchSize(),
                column_names=self.obs_column_names,
            )

            obs_and_x_iter = _ObsAndXIterator(
                X,
                obs_tables_iter,
                var_joinids=self._var_joinids,
                batch_size=self.batch_size,
                encoders=self.obs_encoders,
                stats=self._stats,
                return_sparse_X=self.return_sparse_X,
                use_eager_fetch=self.use_eager_fetch,
            )

            for datum_ in obs_and_x_iter:
                yield datum_

    def __len__(self) -> int:
        self._init()
        assert self._obs_joinids is not None

        return len(self._obs_joinids)

    def __getitem__(self, index: int) -> ObsAndXDatum:
        raise NotImplementedError("IterDataPipe can only be iterated")

    def _build_obs_encoders(self, query: soma.ExperimentAxisQuery) -> Encoders:
        pytorch_logger.debug("Initializing encoders")

        obs = query.obs(column_names=self.obs_column_names).concat()
        encoders = {}
        for col in self.obs_column_names:
            if obs[col].type in (pa.string(), pa.large_string()):
                enc = LabelEncoder()
                enc.fit(obs[col].combine_chunks().unique())
                encoders[col] = enc

        return encoders

    # TODO: This does not work in multiprocessing mode, as child process's stats are not collected
    def stats(self) -> Stats:
        """
        Get data loading stats for this ``ExperimentDataPipe``.

        Returns:
            The ``Stats`` object for this ``ExperimentDataPipe``.

        Lifecycle:
            experimental
        """
        return self._stats

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get the shape of the data that will be returned by this ``ExperimentDataPipe``. This is the number of
        obs (cell) and var (feature) counts in the returned data. If used in multiprocessing mode
        (i.e. DataLoader instantiated with num_workers > 0), the obs (cell) count will reflect the size of the
        partition of the data assigned to the active process.

        Returns:
            A 2-tuple of ``int``s, for obs and var counts, respectively.

        Lifecycle:
            experimental
        """
        self._init()
        assert self._obs_joinids is not None
        assert self._var_joinids is not None

        return len(self._obs_joinids), len(self._var_joinids)

    @property
    def obs_encoders(self) -> Encoders:
        """
        Returns a dictionary of ``sklearn.preprocessing.LabelEncoder`` objects, keyed on ``obs`` column names,
        which were used to encode the ``obs`` column values. These encoders can be used to decode the encoded values as
        follows:

        >>> exp_data_pipe.obs_encoders["<obs_attr_name>"].inverse_transform(encoded_values)

        See https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html.

        Returns:
            A ``Dict[str, LabelEncoder]``, mapping column names to ``LabelEncoder``s.
        """
        self._init()
        assert self._encoders is not None

        return self._encoders


# Note: must be a top-level function (and not a lambda), to play nice with multiprocessing pickling
def _collate_noop(x: Any) -> Any:
    return x


# TODO: Move into somacore.ExperimentAxisQuery
def experiment_dataloader(
    datapipe: pipes.IterDataPipe,
    num_workers: int = 0,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """
    Factory method for PyTorch ``DataLoader``. This method can be used to safely instantiate a
    ``DataLoader`` that works with the ``ExperimentDataPipe``, since some of the ``DataLoader`` constructor params
    are not applicable when using a ``IterDataPipe`` (``batch_size``, ``sampler``, ``batch_sampler``, ``collate_fn``).

    Args:
        datapipe:
            A PyTorch ``IterDataPipe``, which can be an ``ExperimentDataPipe`` or any other ``IterDataPipe`` that has
            been chained to the ``ExperimentDataPipe``.
        num_workers:
            Number of worker processes to use for data loading. If 0, data will be loaded in the main process.
        **dataloader_kwargs:
            Additional keyword arguments to pass to the ``torch.utils.data.DataLoader`` constructor,
            except for ``batch_size``, ``sampler``, ``batch_sampler``, and ``collate_fn``, which are not supported when using ``ExperimentDataPipe``.
            See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.

    Returns:
        A ``torch.utils.data.DataLoader``.

    Raises:
        ValueError: if any of the ``batch_size``, ``sampler``, ``batch_sampler``, or ``collate_fn`` params are passed as keyword arguments.


    Lifecycle:
        experimental
    """

    unsupported_dataloader_args = ["batch_size", "sampler", "batch_sampler", "collate_fn"]
    if set(unsupported_dataloader_args).intersection(dataloader_kwargs.keys()):
        raise ValueError(f"The {','.join(unsupported_dataloader_args)} DataLoader params are not supported")

    if num_workers > 0:
        _init_multiprocessing()

    return DataLoader(
        datapipe,
        batch_size=None,  # batching is handled by our ExperimentDataPipe
        num_workers=num_workers,
        # avoid use of default collator, which adds an extra (3rd) dimension to the tensor batches
        collate_fn=_collate_noop,
        **dataloader_kwargs,
    )


def _init_multiprocessing() -> None:
    """Ensures use of "spawn" for starting child processes with multiprocessing.
    Forked processes are known to be problematic:
      https://pytorch.org/docs/stable/notes/multiprocessing.html#avoiding-and-fighting-deadlocks
    Also, CUDA does not support forked child processes:
      https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing

    """
    torch.multiprocessing.set_start_method("fork", force=True)
    orig_start_method = torch.multiprocessing.get_start_method()
    if orig_start_method != "spawn":
        if orig_start_method:
            pytorch_logger.warning(
                "switching torch multiprocessing start method from "
                f'"{torch.multiprocessing.get_start_method()}" to "spawn"'
            )
        torch.multiprocessing.set_start_method("spawn", force=True)
