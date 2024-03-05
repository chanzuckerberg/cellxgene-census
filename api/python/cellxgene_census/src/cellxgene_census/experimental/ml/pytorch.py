import gc
import logging
import os
from contextlib import contextmanager
from datetime import timedelta
from math import ceil
from time import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
import pyarrow as pa
import scipy
import tiledbsoma as soma
import torch
import torchdata.datapipes.iter as pipes
from attr import define
from numpy.random import Generator
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from ... import get_default_soma_context
from ..util._eager_iter import _EagerIterator

pytorch_logger = logging.getLogger("cellxgene_census.experimental.pytorch")

# TODO: Rename to reflect the correct order of the Tensors within the tuple: (X, obs)
ObsAndXDatum = Tuple[Tensor, Tensor]
"""Return type of ``ExperimentDataPipe`` that pairs a Tensor of ``obs`` row(s) with a Tensor of ``X`` matrix row(s).
The Tensors are rank 1 if ``batch_size`` is 1, otherwise the Tensors are rank 2."""


@define
class _SOMAChunk:
    """Return type of ``_ObsAndXSOMAIterator`` that pairs a chunk of ``obs`` rows with the respective rows from the ``X``
    matrix.

    Lifecycle:
        experimental
    """

    obs: pd.DataFrame
    X: scipy.sparse.spmatrix
    stats: "Stats"

    def __len__(self) -> int:
        return len(self.obs)


Encoders = Dict[str, LabelEncoder]
"""A dictionary of ``LabelEncoder``s keyed by the ``obs`` column name."""


@define
class Stats:
    """Statistics about the data retrieved by ``ExperimentDataPipe`` via SOMA API. This is useful for assessing the read
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

    n_soma_chunks: int = 0
    """The number of chunks retrieved"""

    def __str__(self) -> str:
        return f"{self.n_soma_chunks=}, {self.n_obs=}, {self.nnz=}, " f"elapsed={timedelta(seconds=self.elapsed)}"

    def __add__(self, other: "Stats") -> "Stats":
        self.n_obs += other.n_obs
        self.nnz += other.nnz
        self.elapsed += other.elapsed
        self.n_soma_chunks += other.n_soma_chunks
        return self


@contextmanager
def _open_experiment(
    uri: str,
    aws_region: Optional[str] = None,
) -> soma.Experiment:
    """Internal method for opening a SOMA ``Experiment`` as a context manager."""
    context = get_default_soma_context().replace(tiledb_config={"vfs.s3.region": aws_region} if aws_region else {})

    with soma.Experiment.open(uri, context=context) as exp:
        yield exp


class _ObsAndXSOMAIterator(Iterator[_SOMAChunk]):
    """Iterates the SOMA chunks of corresponding ``obs`` and ``X`` data. This is an internal class,
    not intended for public use.
    """

    X: soma.SparseNDArray
    """A handle to the full X data of the SOMA ``Experiment``"""

    obs_joinids_chunks_iter: Iterator[npt.NDArray[np.int64]]

    var_joinids: npt.NDArray[np.int64]
    """The ``var`` joinids to be retrieved from the SOMA ``Experiment``"""

    def __init__(
        self,
        obs: soma.DataFrame,
        X: soma.SparseNDArray,
        obs_column_names: Sequence[str],
        obs_joinids_chunked: List[npt.NDArray[np.int64]],
        var_joinids: npt.NDArray[np.int64],
        shuffle_rng: Optional[Generator] = None,
    ):
        self.obs = obs
        self.X = X
        self.obs_column_names = obs_column_names
        self.obs_joinids_chunks_iter = self._maybe_local_shuffle_obs_joinids(obs_joinids_chunked, shuffle_rng)
        self.var_joinids = var_joinids

    @staticmethod
    def _maybe_local_shuffle_obs_joinids(
        obs_joinids_chunked: List[npt.NDArray[np.int64]],
        shuffle_rng: Optional[Generator] = None,
    ) -> Iterator[npt.NDArray[np.int64]]:
        return (
            shuffle_rng.permutation(obs_joinid_chunk) if shuffle_rng else obs_joinid_chunk
            for obs_joinid_chunk in obs_joinids_chunked
        )

    def __next__(self) -> _SOMAChunk:
        pytorch_logger.debug("Retrieving next SOMA chunk...")
        start_time = time()

        # If no more chunks to iterate through, raise StopIteration, as all iterators do when at end
        obs_joinids_chunk = next(self.obs_joinids_chunks_iter)

        obs_batch = (
            self.obs.read(
                coords=(obs_joinids_chunk,),
                column_names=self.obs_column_names,
            )
            .concat()
            .to_pandas()
            .set_index("soma_joinid")
        )
        assert obs_batch.shape[0] == obs_joinids_chunk.shape[0]

        # handle case of empty result (first batch has 0 rows)
        if len(obs_batch) == 0:
            raise StopIteration

        # reorder obs rows to match obs_joinids_chunk ordering, which may be shuffled
        obs_batch = obs_batch.reindex(obs_joinids_chunk, copy=False)

        # note: the `blockwise` call is employed for its ability to reindex the axes of the sparse matrix,
        # but the blockwise iteration feature is not used (block_size is set to retrieve the chunk as a single block)
        scipy_iter = (
            self.X.read(coords=(obs_joinids_chunk, self.var_joinids))
            .blockwise(axis=0, size=len(obs_joinids_chunk), eager=False)
            .scipy(compress=True)
        )
        X_batch, _ = next(scipy_iter)
        assert obs_batch.shape[0] == X_batch.shape[0]

        stats = Stats()
        stats.n_obs += X_batch.shape[0]
        stats.nnz += X_batch.nnz
        stats.elapsed += int(time() - start_time)
        stats.n_soma_chunks += 1

        pytorch_logger.debug(f"Retrieved SOMA chunk: {stats}")
        return _SOMAChunk(obs=obs_batch, X=X_batch, stats=stats)


def run_gc() -> Tuple[Tuple[Any, Any, Any], Tuple[Any, Any, Any]]:  # noqa: D103
    proc = psutil.Process(os.getpid())

    pre_gc = proc.memory_full_info(), psutil.virtual_memory(), psutil.swap_memory()
    gc.collect()
    post_gc = proc.memory_full_info(), psutil.virtual_memory(), psutil.swap_memory()

    pytorch_logger.debug(f"gc:  pre={pre_gc}")
    pytorch_logger.debug(f"gc: post={post_gc}")

    return pre_gc, post_gc


class _ObsAndXIterator(Iterator[ObsAndXDatum]):
    """Iterates through a set of ``obs`` and corresponding ``X`` rows, where the rows to be returned are specified by
    the ``obs_tables_iter`` argument. For the specified ``obs` rows, the corresponding ``X`` data is loaded and
    joined together. It is returned from this iterator as 2-tuples of ``X`` and obs Tensors.

    Internally manages the retrieval of data in SOMA-sized chunks, fetching the next chunk of SOMA data as needed.
    Supports fetching the data in an eager manner, where the next SOMA chunk is fetched while the current chunk is
    being read. This is an internal class, not intended for public use.
    """

    soma_chunk_iter: Iterator[_SOMAChunk]
    """The iterator for SOMA chunks of paired obs and X data"""

    soma_chunk: Optional[_SOMAChunk]
    """The current SOMA chunk of obs and X data"""

    i: int = -1
    """Index into current obs ``SOMA`` chunk"""

    def __init__(
        self,
        obs: soma.DataFrame,
        X: soma.SparseNDArray,
        obs_column_names: Sequence[str],
        obs_joinids_chunked: List[npt.NDArray[np.int64]],
        var_joinids: npt.NDArray[np.int64],
        batch_size: int,
        encoders: Dict[str, LabelEncoder],
        stats: Stats,
        return_sparse_X: bool,
        use_eager_fetch: bool,
        shuffle_rng: Optional[Generator] = None,
    ) -> None:
        self.soma_chunk_iter = _ObsAndXSOMAIterator(
            obs, X, obs_column_names, obs_joinids_chunked, var_joinids, shuffle_rng
        )
        if use_eager_fetch:
            self.soma_chunk_iter = _EagerIterator(self.soma_chunk_iter)
        self.soma_chunk = None
        self.var_joinids = var_joinids
        self.batch_size = batch_size
        self.return_sparse_X = return_sparse_X
        self.encoders = encoders
        self.stats = stats
        self.max_process_mem_usage_bytes = 0
        self.X_dtype = X.schema[2].type.to_pandas_dtype()

    def __next__(self) -> ObsAndXDatum:
        """Read the next torch batch, possibly across multiple soma chunks."""
        obs: pd.DataFrame = pd.DataFrame()
        X: sparse.csr_matrix = sparse.csr_matrix((0, len(self.var_joinids)), dtype=self.X_dtype)

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
            data={"soma_joinid": obs.index},
            columns=["soma_joinid"] + obs.columns.tolist(),
            dtype=np.int64,
        )
        # TODO: Encode the entire SOMA chunk at once in _read_partial_torch_batch()
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
        """Reads a torch-size batch of data from the current SOMA chunk, returning a torch-size batch whose size may
        contain fewer rows than the requested ``batch_size``. This can happen when the remaining rows in the current
        SOMA chunk are fewer than the requested ``batch_size``.
        """
        if self.soma_chunk is None or not (0 <= self.i < len(self.soma_chunk)):
            # GC memory from previous soma_chunk
            self.soma_chunk = None
            mem_info = run_gc()
            self.max_process_mem_usage_bytes = max(self.max_process_mem_usage_bytes, mem_info[0][0].uss)

            self.soma_chunk: _SOMAChunk = next(self.soma_chunk_iter)
            self.stats += self.soma_chunk.stats
            self.i = 0

            pytorch_logger.debug(f"Retrieved SOMA chunk totals: {self.stats}")

        obs_batch = self.soma_chunk.obs
        X_batch = self.soma_chunk.X

        safe_batch_size = min(batch_size, len(obs_batch) - self.i)
        slice_ = slice(self.i, self.i + safe_batch_size)
        assert slice_.stop <= obs_batch.shape[0]

        obs_rows = obs_batch.iloc[slice_]
        assert obs_rows.index.is_unique
        assert safe_batch_size == obs_rows.shape[0]

        X_csr_scipy = X_batch[slice_]
        assert obs_rows.shape[0] == X_csr_scipy.shape[0]

        self.i += safe_batch_size

        return obs_rows, X_csr_scipy


class ExperimentDataPipe(pipes.IterDataPipe[Dataset[ObsAndXDatum]]):  # type: ignore
    r"""An :class:`torchdata.datapipes.iter.IterDataPipe` that reads ``obs`` and ``X`` data from a
    :class:`tiledbsoma.Experiment`, based upon the specified queries along the ``obs`` and ``var`` axes. Provides an
    iterator over these data when the object is passed to Python's built-in ``iter`` function.

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

    The ``return_sparse_X`` parameter controls whether the ``X`` data is returned as a dense or sparse
    :class:`torch.Tensor`. If the model supports use of sparse :class:`torch.Tensor`\ s, this will reduce memory usage.

    The ``obs_column_names`` parameter determines the data columns that are returned in the ``obs`` Tensor. The first
    element is always the ``soma_joinid`` of the ``obs`` :class:`pandas.DataFrame` (or, equivalently, the
    ``soma_dim_0`` of the ``X`` matrix). The remaining elements are the ``obs`` columns specified by
    ``obs_column_names``, and string-typed columns are encoded as integer values. If needed, these values can be decoded
    by obtaining the encoder for a given ``obs`` column name and calling its ``inverse_transform`` method:

    >>> exp_data_pipe.obs_encoders["<obs_attr_name>"].inverse_transform(encoded_values)

    Lifecycle:
        experimental
    """

    _initialized: bool

    _obs_joinids: Optional[npt.NDArray[np.int64]]

    _var_joinids: Optional[npt.NDArray[np.int64]]

    _encoders: Optional[Encoders]

    _stats: Stats

    _shuffle_rng: Optional[Generator]

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
        shuffle: bool = False,
        seed: Optional[int] = None,
        return_sparse_X: bool = False,
        soma_chunk_size: Optional[int] = None,
        use_eager_fetch: bool = True,
    ) -> None:
        r"""Construct a new ``ExperimentDataPipe``.

        Args:
            experiment:
                The :class:`tiledbsoma.Experiment` from which to read data.
            measurement_name:
                The name of the :class:`tiledbsoma.Measurement` to read. Defaults to ``"raw"``.
            X_name:
                The name of the X layer to read. Defaults to ``"X"``.
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
                The number of rows of ``obs`` and ``X`` data to return in each iteration. Defaults to ``1``. A value of
                ``1`` will result in :class:`torch.Tensor` of rank 1 being returns (a single row); larger values will
                result in :class:`torch.Tensor`\ s of rank 2 (multiple rows).
            shuffle:
                Whether to shuffle the ``obs`` and ``X`` data being returned. Defaults to ``False`` (no shuffling).
                For performance reasons, shuffling is performed in two steps: 1) a global shuffling, where contiguous
                rows are grouped into chunks and the order of the chunks is randomized, and then 2) a local
                shuffling, where the rows within each chunk are shuffled. Since this class must retrieve data
                in chunks (to keep memory requirements to a fixed size), global shuffling ensures that a given row in
                the shuffled result can originate from any position in the non-shuffled result ordering. If shuffling
                only occurred within each chunk (i.e. "local" shuffling), the first chunk's rows would always be
                returned first, the second chunk's rows would always be returned second, and so on. The chunk size is
                determined by the ``soma_chunk_size`` parameter. Note that rows within a chunk will maintain
                proximity, even after shuffling, so some experimentation may be required to ensure the shuffling is
                sufficient for the model training process. To this end, the ``soma_chunk_size`` can be treated as a
                hyperparameter that can be tuned.
            seed:
                The random seed used for shuffling. Defaults to ``None`` (no seed). This *must* be specified when using
                :class:`torch.nn.parallel.DistributedDataParallel` to ensure data partitions are disjoint across worker
                processes.
            return_sparse_X:
                Controls whether the ``X`` data is returned as a dense or sparse :class:`torch.Tensor`. As ``X`` data is
                very sparse, setting this to ``True`` will reduce memory usage, if the model supports use of sparse
                :class:`torch.Tensor`\ s. Defaults to ``False``, since sparse :class:`torch.Tensor`\ s are still
                experimental in PyTorch.
            soma_chunk_size:
                The number of ``obs``/``X`` rows to retrieve when reading data from SOMA. This impacts two aspects of
                this class's behavior: 1) The maximum memory utilization, with larger values providing
                better read performance, but also requiring more memory; 2) The granularity of the global shuffling
                step (see ``shuffle`` parameter for details). If not specified, the value is set to utilize ~1 GiB of
                RAM per SOMA chunk read, based upon the number of ``var`` columns (cells/features) being requested
                and assuming X data sparsity of 95%; the number of rows per chunk will depend on the number of
                ``var`` columns being read.
            use_eager_fetch:
                Fetch the next SOMA chunk of ``obs`` and ``X`` data immediately after a previously fetched SOMA chunk is made
                available for processing via the iterator. This allows network (or filesystem) requests to be made in
                parallel with client-side processing of the SOMA data, potentially improving overall performance at the
                cost of doubling memory utilization. Defaults to ``True``.

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
        self.soma_chunk_size = soma_chunk_size
        self.use_eager_fetch = use_eager_fetch
        self._stats = Stats()
        self._encoders = None
        self._obs_joinids = None
        self._var_joinids = None
        self._shuffle_rng = np.random.default_rng(seed) if shuffle else None
        self._initialized = False

        if "soma_joinid" not in self.obs_column_names:
            self.obs_column_names = ["soma_joinid", *self.obs_column_names]

    def _init(self) -> None:
        if self._initialized:
            return

        pytorch_logger.debug("Initializing ExperimentDataPipe")

        with _open_experiment(self.exp_uri, self.aws_region) as exp:
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
    def _subset_ids_to_partition(
        ids_chunked: List[npt.NDArray[np.int64]],
        partition_index: int,
        num_partitions: int,
    ) -> List[npt.NDArray[np.int64]]:
        """Returns a single partition of the obs_joinids_chunked (a 2D ndarray), based upon the current process's distributed rank and world
        size.
        """
        # subset to a single partition
        # typing does not reflect that is actually a List of 2D NDArrays
        partition_indices = np.array_split(range(len(ids_chunked)), num_partitions)
        partition = [ids_chunked[i] for i in partition_indices[partition_index]]

        if pytorch_logger.isEnabledFor(logging.DEBUG) and len(partition) > 0:
            pytorch_logger.debug(
                f"Process {os.getpid()} handling partition {partition_index + 1} of {num_partitions}, "
                f"partition_size={sum([len(chunk) for chunk in partition])}"
            )

        return partition

    @staticmethod
    def _compute_partitions(
        loader_partition: int,
        loader_partitions: int,
        dist_partition: int,
        num_dist_partitions: int,
    ) -> Tuple[int, int]:
        # NOTE: Can alternately use a `worker_init_fn` to split among workers split workload
        total_partitions = num_dist_partitions * loader_partitions
        partition = dist_partition * loader_partitions + loader_partition
        return partition, total_partitions

    def __iter__(self) -> Iterator[ObsAndXDatum]:
        self._init()
        assert self._obs_joinids is not None
        assert self._var_joinids is not None

        if self.soma_chunk_size is None:
            # set soma_chunk_size to utilize ~1 GiB of RAM per SOMA chunk; assumes 95% X data sparsity, 8 bytes for the
            # X value and 8 bytes for the sparse matrix indices, and a 100% working memory overhead (2x).
            X_row_memory_size = 0.05 * len(self._var_joinids) * 8 * 3 * 2
            self.soma_chunk_size = int((1 * 1024**3) / X_row_memory_size)
        pytorch_logger.debug(f"Using {self.soma_chunk_size=}")

        if (
            self.return_sparse_X
            and torch.utils.data.get_worker_info()
            and torch.utils.data.get_worker_info().num_workers > 0
        ):
            raise NotImplementedError(
                "torch does not work with sparse tensors in multi-processing mode "
                "(see https://github.com/pytorch/pytorch/issues/20248)"
            )

        # chunk the obs joinids into batches of size soma_chunk_size
        obs_joinids_chunked = self._chunk_ids(self._obs_joinids, self.soma_chunk_size)

        # globally shuffle the chunks, if requested
        if self._shuffle_rng:
            self._shuffle_rng.shuffle(obs_joinids_chunked)

        # subset to a single partition, as needed for distributed training and multi-processing datat loading
        worker_info = torch.utils.data.get_worker_info()
        partition, partitions = self._compute_partitions(
            loader_partition=worker_info.id if worker_info else 0,
            loader_partitions=worker_info.num_workers if worker_info else 1,
            dist_partition=dist.get_rank() if dist.is_initialized() else 0,
            num_dist_partitions=dist.get_world_size() if dist.is_initialized() else 1,
        )
        obs_joinids_chunked_partition: List[npt.NDArray[np.int64]] = self._subset_ids_to_partition(
            obs_joinids_chunked, partition, partitions
        )

        with _open_experiment(self.exp_uri, self.aws_region) as exp:
            obs_and_x_iter = _ObsAndXIterator(
                obs=exp.obs,
                X=exp.ms[self.measurement_name].X[self.layer_name],
                obs_column_names=self.obs_column_names,
                obs_joinids_chunked=obs_joinids_chunked_partition,
                var_joinids=self._var_joinids,
                batch_size=self.batch_size,
                encoders=self.obs_encoders,
                stats=self._stats,
                return_sparse_X=self.return_sparse_X,
                use_eager_fetch=self.use_eager_fetch,
                shuffle_rng=self._shuffle_rng,
            )

            yield from obs_and_x_iter

            pytorch_logger.debug(
                "max process memory usage=" f"{obs_and_x_iter.max_process_mem_usage_bytes / (1024 ** 3):.3f} GiB"
            )

    @staticmethod
    def _chunk_ids(ids: npt.NDArray[np.int64], chunk_size: int) -> List[npt.NDArray[np.int64]]:
        num_chunks = max(1, ceil(len(ids) / chunk_size))
        pytorch_logger.debug(f"Shuffling {len(ids)} obs joinids into {num_chunks} chunks of {chunk_size}")
        return np.array_split(ids, num_chunks)

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
        """Get data loading stats for this :class:`cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe`.

        Returns:
            The :class:`cellxgene_census.experimental.ml.pytorch.Stats` object for this
            :class:`cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe`.

        Lifecycle:
            experimental
        """
        return self._stats

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the data that will be returned by this :class:`cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe`.
        This is the number of obs (cell) and var (feature) counts in the returned data. If used in multiprocessing mode
        (i.e. :class:`torch.utils.data.DataLoader` instantiated with num_workers > 0), the obs (cell) count will reflect
        the size of the partition of the data assigned to the active process.

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
        """Returns a dictionary of :class:`sklearn.preprocessing.LabelEncoder` objects, keyed on ``obs`` column names,
        which were used to encode the ``obs`` column values.

        These encoders can be used to decode the encoded values as follows:

        >>> exp_data_pipe.obs_encoders["<obs_attr_name>"].inverse_transform(encoded_values)

        Returns:
            A ``Dict[str, LabelEncoder]``, mapping column names to :class:`sklearn.preprocessing.LabelEncoder` objects.
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
    """Factory method for :class:`torch.utils.data.DataLoader`. This method can be used to safely instantiate a
    :class:`torch.utils.data.DataLoader` that works with :class:`cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe`,
    since some of the :class:`torch.utils.data.DataLoader` constructor parameters are not applicable when using a
    :class:`torchdata.datapipes.iter.IterDataPipe` (``shuffle``, ``batch_size``, ``sampler``, ``batch_sampler``,
    ``collate_fn``).

    Args:
        datapipe:
            An :class:`torchdata.datapipes.iter.IterDataPipe`, which can be an
            :class:`cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe` or any other
            :class:`torchdata.datapipes.iter.IterDataPipe` that has been chained to the
            :class:`cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe`.
        num_workers:
            Number of worker processes to use for data loading. If ``0``, data will be loaded in the main process.
        **dataloader_kwargs:
            Additional keyword arguments to pass to the :class:`torch.utils.data.DataLoader` constructor,
            except for ``shuffle``, ``batch_size``, ``sampler``, ``batch_sampler``, and ``collate_fn``, which are not
            supported when using :class:`cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe`.

    Returns:
        A :class:`torch.utils.data.DataLoader`.

    Raises:
        ValueError: if any of the ``shuffle``, ``batch_size``, ``sampler``, ``batch_sampler``, or ``collate_fn`` params
            are passed as keyword arguments.

    Lifecycle:
        experimental
    """
    unsupported_dataloader_args = [
        "shuffle",
        "batch_size",
        "sampler",
        "batch_sampler",
        "collate_fn",
    ]
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
        # shuffling is handled by our ExperimentDataPipe
        shuffle=False,
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
