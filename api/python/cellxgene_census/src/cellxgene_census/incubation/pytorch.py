import sys
from datetime import timedelta
from time import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import tiledbsoma as soma
import torch
from attr import attrs
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from somacore import BatchSize
from somacore.query import AxisQuery, _fast_csr
from tiledbsoma._read_iters import TableReadIter
from torch import Tensor
from torch.utils.data import DataLoader, IterDataPipe
from torch.utils.data.dataset import Dataset

ObsDatum = Tuple[Tensor, torch.sparse_coo_tensor]

DEFAULT_BUFFER_BYTES = 1024**3

DEFAULT_WORKER_TIMEOUT = 120


@attrs
class Stats:
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


# wrap in dataset iterator
class _ObsAndXIterator(Iterator[ObsDatum]):
    """
    Iterates through a set of obs and related X rows, specified as `soma_joinid`s. Encapsulates the batch-based data
    fetching from TileDB-SOMA objects, providing row-based iteration.
    """

    obs_tables_iter: TableReadIter
    """Iterates the TileDB-SOMA batches (tables) of obs data"""

    obs_batch_: pd.DataFrame = pd.DataFrame()
    """The current TileDB-SOMA batch of obs data"""

    X_batch: scipy.matrix = None
    """All X data for the soma_joinids of the current obs - batch"""

    i: int = -1
    """Index into current obs TileDB-SOMA batch"""

    def __init__(
        self,
        obs_joinids: pa.Array,
        var_joinids: pa.Array,
        exp: soma.Experiment,
        measurement_name: str,
        X_layer_name: str,
        batch_size: int,
        encoders: Dict[str, LabelEncoder],
        stats: Stats,
        obs_column_names: Sequence[str],
        dense_X: bool,
    ) -> None:
        self.obs_joinids = obs_joinids
        self.var_joinids = var_joinids
        self.exp = exp
        self.X: soma.SparseNDArray = exp.ms[measurement_name].X[X_layer_name]
        self.batch_size = batch_size
        self.dense_X = dense_X
        self.obs_tables_iter = exp.obs.read(
            coords=(obs_joinids,), batch_size=BatchSize(), column_names=obs_column_names
        )
        self.encoders = encoders
        self.stats = stats

    def __next__(self) -> ObsDatum:
        # read the torch batch, possibly across multiple soma batches
        obs: pd.DataFrame = pd.DataFrame()
        X: sparse.csr_matrix = sparse.csr_matrix((0, len(self.var_joinids)))

        while len(obs) < self.batch_size:
            try:
                obs_partial, X_partial = self._read_partial_torch_batch(self.batch_size)
                if X is None:
                    obs = obs_partial
                    X = X_partial
                else:
                    obs = pd.concat([obs, obs_partial], axis=0)
                    X = sparse.vstack([X, X_partial])

            except StopIteration:
                break

        if len(obs) == 0:
            raise StopIteration

        obs_encoded = pd.DataFrame(
            data={"soma_joinid": obs.soma_joinid}, columns=obs.columns, dtype=np.int32, index=obs.index
        )
        for col, enc in self.encoders.items():
            obs_encoded[col] = enc.transform(obs[col])

        obs_tensor = torch.Tensor(obs_encoded.to_numpy()).int()
        if self.dense_X:
            return obs_tensor, torch.Tensor(X.todense())
        else:
            coo = X.tocoo()

            X_tensor = torch.sparse_coo_tensor(
                # Note: The `np.array` seems unnecessary, but PyTorch warns bare array is "extremely slow"
                indices=torch.Tensor(np.array([coo.row, coo.col])),
                values=coo.data,
                size=coo.shape,
            )

            if self.batch_size == 1:
                obs_tensor = obs_tensor[0]
                X_tensor = X_tensor[0]

            return obs_tensor, X_tensor

    def _read_partial_torch_batch(self, batch_size: int) -> Tuple[pd.DataFrame, sparse.csr_matrix]:
        safe_batch_size = min(batch_size, len(self.obs_batch) - self.i)
        slice_ = slice(self.i, self.i + safe_batch_size)
        obs_rows = self.obs_batch.iloc[slice_]
        X_csr_scipy = self.X_batch[slice_]
        self.i += safe_batch_size
        return obs_rows, X_csr_scipy

    @property
    # TODO: Retrieve next batch asynchronously, so it's available before the current batch's iteration ends
    def obs_batch(self) -> pd.DataFrame:
        """
        Returns the current SOMA batch of obs rows.
        If the current SOMA batch has been fully iterated, loads the next SOMA batch of both obs and X data and returns
        the new obs batch (only).
        Raises StopIteration if there are no more SOMA batches to retrieve.
        """
        if 0 <= self.i < len(self.obs_batch_):
            return self.obs_batch_

        print("Retrieving next TileDB-SOMA batch...")
        start_time = time()
        # If no more batch to iterate through, raise StopIteration, as all iterators do when at end
        obs_table = next(self.obs_tables_iter)
        self.obs_batch_ = obs_table.to_pandas()
        # handle case of empty result (first batch has 0 rows)
        if len(self.obs_batch_) == 0:
            raise StopIteration
        self.X_batch = _fast_csr.read_scipy_csr(self.X, obs_table["soma_joinid"].combine_chunks(), self.var_joinids)
        self.i = 0
        self.stats.n_obs += self.X_batch.shape[0]
        self.stats.nnz += self.X_batch.nnz
        self.stats.elapsed = int(self.stats.elapsed + (time() - start_time))
        self.stats.n_soma_batches += 1
        print(f"Retrieved batch: shape={self.X_batch.shape}, {self.stats}")
        return self.obs_batch_


class ExperimentDataPipe(IterDataPipe[Dataset[ObsDatum]]):  # type: ignore
    """
    An iterable-style data pipe.
    """

    def __getitem__(self, index: int) -> ObsDatum:
        raise NotImplementedError("IterDataPipe can only be iterated")

    _obs_and_x_iter: Optional[_ObsAndXIterator]

    _encoders: Dict[str, LabelEncoder]

    _stats: Stats

    def __init__(
        self,
        exp_uri: str,
        ms_name: str,
        layer_name: str,
        obs_query: Optional[AxisQuery] = None,
        var_query: Optional[AxisQuery] = None,
        obs_column_names: Sequence[str] = (),
        batch_size: int = 1,
        dense_X: bool = False,
        num_workers: int = 0,
        buffer_bytes: int = 1024**3,
    ) -> None:
        if num_workers > 1 and not dense_X:
            raise NotImplementedError(
                "torch does not work with sparse tensors in multi-processing mode "
                "(see https://github.com/pytorch/pytorch/issues/20248)"
            )

        self.exp_uri = exp_uri
        self.ms_name = ms_name
        self.layer_name = layer_name
        self.obs_query = obs_query
        self.var_query = var_query
        self.obs_column_names = obs_column_names
        self.batch_size = batch_size
        self.dense_X = dense_X
        self.buffer_bytes = buffer_bytes
        self._stats = Stats()
        self._encoders = {}

        if "soma_joinid" not in self.obs_column_names:
            self.obs_column_names = ["soma_joinid", *self.obs_column_names]

    def __iter__(self) -> Iterator[ObsDatum]:
        # TODO: support multiple layers, per somacore.query.query.ExperimentAxisQuery.to_anndata()

        # TODO: handle closing of exp when iterator is no longer in use; may need be used as a ContextManager,
        #  but not clear how we can do that when used by DataLoader
        # TODO: for dev-time use, specify smaller batch size via platform_config, once supported
        context = soma.options.SOMATileDBContext().replace(
            tiledb_config={"py.init_buffer_bytes": self.buffer_bytes, "soma.init_buffer_bytes": self.buffer_bytes}
        )

        exp = soma.Experiment.open(self.exp_uri, platform_config={}, context=context)
        query = exp.axis_query(
            measurement_name=self.ms_name,
            obs_query=self.obs_query,
            var_query=self.var_query,
        )

        obs_joinids = query.obs_joinids()
        var_joinids = query.var_joinids()

        obs = query.obs(column_names=self.obs_column_names).concat()
        for col in self.obs_column_names:
            if obs[col].type in (pa.string(), pa.large_string()):
                enc = LabelEncoder()
                enc.fit(obs[col].combine_chunks().unique())
                self._encoders[col] = enc

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            partition, partitions = worker_info.id, worker_info.num_workers
        else:
            partition, partitions = 0, 1

        if partitions is not None:
            # partitioned data loading
            # NOTE: Can alternately use a `worker_init_fn` to split among workers split workload
            partition_size = len(obs_joinids) // partitions
            partition_start = partition_size * partition
            partition_end_excl = min(len(obs_joinids), partition_start + partition_size)
            obs_joinids = obs_joinids[partition_start:partition_end_excl]
            print(
                f"Partition {partition + 1} of {partitions}, range={partition_start}:{partition_end_excl}, "
                f"{partition_size:}"
            )

        self._obs_and_x_iter = _ObsAndXIterator(
            obs_joinids=obs_joinids,
            var_joinids=var_joinids,
            exp=exp,
            measurement_name=self.ms_name,
            X_layer_name=self.layer_name,
            batch_size=self.batch_size,
            obs_column_names=self.obs_column_names,
            dense_X=self.dense_X,
            encoders=self._encoders,
            stats=self._stats,
        )
        return self._obs_and_x_iter

    def obs_encoders(self) -> Dict[str, LabelEncoder]:
        return self._encoders

    def stats(self) -> Stats:
        return self._stats


# Note: must be a top-level function (and not a lambda), to place nice with multiprocessing (pickling)
def collate_noop(x: Any) -> Any:
    return x


# TODO: Move into somacore.ExperimentAxisQuery
def experiment_dataloader(
    exp_uri: str,
    ms_name: str,
    layer_name: str,
    obs_query: Optional[AxisQuery] = None,
    var_query: Optional[AxisQuery] = None,
    obs_column_names: Sequence[str] = (),
    dense_X: bool = False,
    batch_size: int = 1,
    num_workers: int = 1,
    buffer_bytes: int = DEFAULT_BUFFER_BYTES,
    **dataloader_kwargs: Dict[str, Any],
) -> Tuple[DataLoader, Stats]:
    """
    Factory method for PyTorch DataLoader. Provides a safer, more convenient interface for instantiating a DataLoader
    that works with the ExperimentDataPipe, since not all of DataLoader's params can be used (batch_size, samples,
    batch_sampler, collate_fn).
    """

    unsupported_dataloader_args = ["sampler", "batch_sampler", "collate_fn"]
    if set(unsupported_dataloader_args).intersection(dataloader_kwargs.keys()):
        raise ValueError(f"The {','.join(unsupported_dataloader_args)} DataLoader params are not supported")

    exp_datapipe = ExperimentDataPipe(
        exp_uri=exp_uri,
        obs_query=obs_query,
        var_query=var_query,
        ms_name=ms_name,
        layer_name=layer_name,
        dense_X=dense_X,
        obs_column_names=obs_column_names,
        batch_size=batch_size,
        num_workers=num_workers,
        buffer_bytes=buffer_bytes,
    )

    if "batch_size" in dataloader_kwargs:
        del dataloader_kwargs["batch_size"]
    return (
        DataLoader(
            exp_datapipe,
            batch_size=None,
            # avoid use of default collator, which adds an extra (3rd) dimension to the tensor batches
            collate_fn=collate_noop,
            **dataloader_kwargs,
        ),
        exp_datapipe.stats(),
    )


# For testing only
if __name__ == "__main__":
    import tiledbsoma as soma

    (
        soma_experiment_uri,
        measurement_name,
        layer_name_arg,
        obs_value_filter_arg,
        column_names_arg,
        dense_X_arg,
        torch_batch_size_arg,
        num_workers_arg,
    ) = sys.argv[1:]
    dl, stats = experiment_dataloader(
        exp_uri=soma_experiment_uri,
        ms_name=measurement_name,
        layer_name=layer_name_arg,
        obs_query=AxisQuery(value_filter=(obs_value_filter_arg or None)),
        var_query=AxisQuery(coords=(slice(1, 9),)),
        obs_column_names=column_names_arg.split(","),
        dense_X=dense_X_arg.lower() == "dense",
        batch_size=int(torch_batch_size_arg),
        num_workers=int(num_workers_arg),
        buffer_bytes=2**12,
    )

    i = 0
    datum = None
    for i, datum in enumerate(dl):
        if (i + 1) % 1000 == 0:
            print(f"Received {i} torch batches, {stats}:\n{datum}")
    print(f"Received {i} torch batches, {stats}:\n{datum}")
