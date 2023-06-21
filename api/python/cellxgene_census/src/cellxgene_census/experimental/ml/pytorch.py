import logging
import os
import sys
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
from attr import attrs
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from somacore.query import _fast_csr
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import cellxgene_census
from cellxgene_census._open import _build_soma_tiledb_context

ObsDatum = Tuple[Tensor, Tensor]

Encoders = Dict[str, LabelEncoder]

pytorch_logger = logging.getLogger("cellxgene_census.experimental.pytorch")
pytorch_logger.setLevel(logging.INFO)


@attrs
class Stats:
    """
    Statistics about the data retrieved by ``ExperimentDataPipe`` via SOMA API.

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


def _open_experiment(
    uri: str, aws_region: Optional[str] = None, soma_buffer_bytes: Optional[int] = None
) -> soma.Experiment:
    context = _build_soma_tiledb_context(aws_region)

    if soma_buffer_bytes is not None:
        context = context.replace(
            tiledb_config={
                "py.init_buffer_bytes": soma_buffer_bytes,
                "soma.init_buffer_bytes": soma_buffer_bytes,
            }
        )

    return soma.Experiment.open(uri, context=context)


class _ObsAndXIterator(Iterator[ObsDatum]):
    """
    Iterates through a set of obs and related X rows, specified as ``soma_joinid``s. Encapsulates the batch-based data
    fetching from SOMA objects, providing row-based iteration.
    """

    obs_tables_iter: somacore.ReadIter[pa.Table]
    """Iterates the SOMA batches (tables) of obs data"""

    obs_batch_: pd.DataFrame = pd.DataFrame()
    """The current SOMA batch of obs data"""

    X_batch: scipy.matrix = None
    """All X data for the ``soma_joinid``s of the current obs - batch"""

    i: int = -1
    """Index into current obs ``SOMA`` batch"""

    def __init__(
        self,
        obs_joinids: npt.NDArray[np.int64],
        var_joinids: npt.NDArray[np.int64],
        exp_uri: str,
        aws_region: Optional[str],
        measurement_name: str,
        X_layer_name: str,
        batch_size: int,
        encoders: Dict[str, LabelEncoder],
        stats: Stats,
        obs_column_names: Sequence[str],
        sparse_X: bool,
        soma_buffer_bytes: Optional[int] = None,
    ) -> None:
        self.var_joinids = var_joinids
        self.batch_size = batch_size
        self.sparse_X = sparse_X

        # holding reference to SOMA object prevents it from being closed
        self.exp = _open_experiment(exp_uri, aws_region, soma_buffer_bytes=soma_buffer_bytes)
        self.X: soma.SparseNDArray = self.exp.ms[measurement_name].X[X_layer_name]
        self.obs_tables_iter = self.exp.obs.read(
            coords=(obs_joinids,), batch_size=somacore.BatchSize(), column_names=obs_column_names
        )
        self.encoders = encoders
        self.stats = stats

    def __next__(self) -> ObsDatum:
        # read the next torch batch, possibly across multiple soma batches
        obs: pd.DataFrame = pd.DataFrame()
        X: sparse.csr_matrix = sparse.csr_matrix((0, len(self.var_joinids)))

        while len(obs) < self.batch_size:
            try:
                obs_partial, X_partial = self._read_partial_torch_batch(self.batch_size)
                obs = pd.concat([obs, obs_partial], axis=0)
                X = sparse.vstack([X, X_partial])
            except StopIteration:
                break

        if len(obs) == 0:
            raise StopIteration

        obs_encoded = pd.DataFrame(
            data={"soma_joinid": obs.soma_joinid}, columns=obs.columns, dtype=np.int64, index=obs.index
        )
        for col, enc in self.encoders.items():
            obs_encoded[col] = enc.transform(obs[col])

        # `to_numpy()` avoids copying the numpy array data
        obs_tensor = torch.from_numpy(obs_encoded.to_numpy())

        if not self.sparse_X:
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

    def _read_partial_torch_batch(self, batch_size: int) -> Tuple[pd.DataFrame, sparse.csr_matrix]:
        safe_batch_size = min(batch_size, len(self.obs_batch) - self.i)
        slice_ = slice(self.i, self.i + safe_batch_size)
        assert slice_.stop <= self.obs_batch_.shape[0]

        obs_rows = self.obs_batch.iloc[slice_]
        assert obs_rows["soma_joinid"].is_unique
        X_csr_scipy = self.X_batch[slice_]
        assert safe_batch_size == obs_rows.shape[0]
        assert obs_rows.shape[0] == X_csr_scipy.shape[0]

        self.i += safe_batch_size

        return obs_rows, X_csr_scipy

    @property
    # TODO: Retrieve next batch asynchronously, so it's available before the current batch's iteration ends
    def obs_batch(self) -> pd.DataFrame:
        """
        Returns the current SOMA batch of obs rows.
        If the current SOMA batch has been fully iterated, loads the next SOMA batch of both obs and X data and returns
        the new obs batch (only).
        Raises ``StopIteration`` if there are no more SOMA batches to retrieve.
        """
        if 0 <= self.i < len(self.obs_batch_):
            return self.obs_batch_

        pytorch_logger.debug("Retrieving next TileDB-SOMA batch...")
        start_time = time()
        # If no more batches to iterate through, raise StopIteration, as all iterators do when at end
        obs_table = next(self.obs_tables_iter)
        self.obs_batch_ = obs_table.to_pandas()
        # handle case of empty result (first batch has 0 rows)
        if len(self.obs_batch_) == 0:
            raise StopIteration
        self.X_batch = _fast_csr.read_scipy_csr(self.X, obs_table["soma_joinid"].combine_chunks(), self.var_joinids)
        assert self.obs_batch_.shape[0] == self.X_batch.shape[0]
        self.i = 0
        self.stats.n_obs += self.X_batch.shape[0]
        self.stats.nnz += self.X_batch.nnz
        self.stats.elapsed += int(time() - start_time)
        self.stats.n_soma_batches += 1
        # TODO: also show per-batch stats (non-cumulative)
        pytorch_logger.debug(f"Retrieved batch: shape={self.X_batch.shape}, cum_stats: {self.stats}")
        return self.obs_batch_


class ExperimentDataPipe(pipes.IterDataPipe[Dataset[ObsDatum]]):  # type: ignore
    """
    An iterable-style PyTorch ``DataPipe`` that reads obs and X data from a SOMA Experiment, and returns an iterator of
    tuples of PyTorch ``Tensor`` objects.

    >>> (tensor([0., 0., 0., 0., 0., 1., 0., 0., 0.]),  # X data
        tensor([2415,    0,    0], dtype=torch.int64)) # obs data, encoded

    Supports batching via `batch_size` param:

    >>> DataLoader(..., batch_size=3, ...):
        (tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0.],     # X batch
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
         tensor([[2415,    0,    0],                       # obs batch
                 [2416,    0,    4],
                 [2417,    0,    3]], dtype=torch.int64))

    Obs attribute values are encoded as categoricals. Values can be decoded by obtaining the encoder for a given
    attribute:

    >>> exp_data_pipe.obs_encoders()["<obs_attr_name>"].inverse_transform(encoded_values)

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
        sparse_X: bool = False,
        soma_buffer_bytes: Optional[int] = None,
    ) -> None:
        """
        Construct a new ``ExperimentDataPipe``.

        Returns:
            ``ExperimentDataPipe``.

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
        self.sparse_X = sparse_X
        # TODO: This will control the SOMA batch sizes, and could be replaced with a row count once TileDB-SOMA
        #  supports `batch_size` param. It affects both the obs and X read operations.
        self.soma_buffer_bytes = soma_buffer_bytes
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

        # TODO: support multiple layers, per somacore.query.query.ExperimentAxisQuery.to_anndata()
        # TODO: handle closing of `_query` (and transitively, `exp`) when iterator is no longer in use; may need be used as a ContextManager,
        #  but not clear how we can do that when used by DataLoader
        exp = _open_experiment(self.exp_uri, self.aws_region, soma_buffer_bytes=self.soma_buffer_bytes)

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

        partition_size = len(ids) // num_partitions
        partition_start = partition_size * partition
        partition_end_excl = min(len(ids), partition_start + partition_size)
        ids = ids[partition_start:partition_end_excl]

        if pytorch_logger.isEnabledFor(logging.DEBUG):
            if len(ids) > 0:
                joinids_start = ids[0]
                joinids_end = ids[-1]
            else:
                joinids_start = joinids_end = 0

            pytorch_logger.debug(
                f"Process {os.getpid()} handling partition {partition + 1} of {num_partitions}, "
                f"index range={partition_start}:{partition_end_excl}, "
                f"soma_joinid range={joinids_start}:{joinids_end}, "
                f"{partition_size:}"
            )

        return ids

    def __iter__(self) -> Iterator[ObsDatum]:
        self._init()
        assert self._obs_joinids is not None
        assert self._var_joinids is not None

        if self.sparse_X and torch.utils.data.get_worker_info() and torch.utils.data.get_worker_info().num_workers > 0:
            raise NotImplementedError(
                "torch does not work with sparse tensors in multi-processing mode "
                "(see https://github.com/pytorch/pytorch/issues/20248)"
            )

        return _ObsAndXIterator(
            obs_joinids=self._partition_obs_joinids(self._obs_joinids),
            var_joinids=self._var_joinids,
            exp_uri=self.exp_uri,
            aws_region=self.aws_region,
            measurement_name=self.measurement_name,
            X_layer_name=self.layer_name,
            batch_size=self.batch_size,
            encoders=self.obs_encoders,
            stats=self._stats,
            obs_column_names=self.obs_column_names,
            sparse_X=self.sparse_X,
            soma_buffer_bytes=self.soma_buffer_bytes,
        )

    def __len__(self) -> int:
        self._init()
        assert self._obs_joinids is not None

        return len(self._obs_joinids)

    def __getitem__(self, index: int) -> ObsDatum:
        raise NotImplementedError("IterDataPipe can only be iterated")

    def _build_obs_encoders(self, query: soma.ExperimentAxisQuery) -> Encoders:
        """
        Returns the encoders that were used to encode obs column values and that are needed to decode them.

        Returns:
            ``Dict[str, LabelEncoder]`` mapping column names to ``LabelEncoder``s.

        Lifecycle:
            experimental
        """
        pytorch_logger.debug("Initializing encoders")

        obs = query.obs(column_names=self.obs_column_names).concat()
        encoders = {}
        for col in self.obs_column_names:
            if obs[col].type in (pa.string(), pa.large_string()):
                enc = LabelEncoder()
                enc.fit(obs[col].combine_chunks().unique())
                encoders[col] = enc

        return encoders

    def stats(self) -> Stats:
        """
        Get data loading stats for this ``ExperimentDataPipe``.

        Args: None.

        Returns:
            ``Stats`` object.

        Lifecycle:
            experimental
        """
        return self._stats

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get the shape of the data that will be returned by this ExperimentDataPipe. This is the number of
        obs (cell) and var (feature) counts in the returned data. If used in multiprocessing mode
        (i.e. DataLoader instantiated with num_workers > 0), the obs (cell) count will reflect the size of the
        partition of the data assigned to the active process.

        Returns:
            2-tuple of ``int``s, for obs and var counts, respectively.

        Lifecycle:
            experimental
        """
        self._init()
        assert self._obs_joinids is not None
        assert self._var_joinids is not None

        return len(self._obs_joinids), len(self._var_joinids)

    @property
    def obs_encoders(self) -> Encoders:
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
    ``DataLoader`` that works with the ``ExperimentDataPipe``, since not all of the ``DataLoader`` constructor params
    can be used (``batch_size``, ``sampler``, ``batch_sampler``, ``collate_fn``).

    Returns:
        PyTorch ``DataLoader``.

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


# For testing only
if __name__ == "__main__":
    import tiledbsoma as soma

    (
        census_uri_arg,
        organism_arg,
        measurement_name_arg,
        layer_name_arg,
        obs_value_filter_arg,
        column_names_arg,
        sparse_X_arg,
        torch_batch_size_arg,
        num_workers_arg,
    ) = sys.argv[1:]

    census = cellxgene_census.open_soma(uri=census_uri_arg)

    exp_datapipe = ExperimentDataPipe(
        experiment=census["census_data"]["homo_sapiens"],
        measurement_name=measurement_name_arg,
        X_name=layer_name_arg,
        obs_query=soma.AxisQuery(value_filter=(obs_value_filter_arg or None)),
        var_query=soma.AxisQuery(coords=(slice(1, 9),)),
        obs_column_names=column_names_arg.split(","),
        batch_size=int(torch_batch_size_arg),
        soma_buffer_bytes=2**12,
        sparse_X=sparse_X_arg.lower() == "sparse",
    )

    dp_shuffle = exp_datapipe.shuffle(buffer_size=len(exp_datapipe))
    dp_train, dp_test = dp_shuffle.random_split(weights={"train": 0.7, "test": 0.3}, seed=1234)

    dl = experiment_dataloader(dp_train, num_workers=int(num_workers_arg))

    i = 0
    datum = None
    for i, datum in enumerate(dl):
        if (i + 1) % 1000 == 0:
            print(f"Received {i} torch batches, {exp_datapipe.stats()}:\n{datum}")
    print(f"Received {i} torch batches, {exp_datapipe.stats()}:\n{datum}")
