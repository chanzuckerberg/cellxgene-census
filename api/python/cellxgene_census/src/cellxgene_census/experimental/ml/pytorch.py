import logging
import os
import sys
from datetime import timedelta
from time import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
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

ObsDatum = Tuple[Tensor, torch.sparse_coo_tensor]

Encoders = Dict[str, LabelEncoder]

DEFAULT_BUFFER_BYTES = 1024**3

DEFAULT_WORKER_TIMEOUT = 120

pytorch_logger = logging.getLogger("cellxgene_census.experimental.pytorch")
pytorch_logger.setLevel(logging.INFO)


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


def _open_experiment(
    uri: str, aws_region: Optional[str] = None, buffer_bytes: int = DEFAULT_BUFFER_BYTES
) -> soma.Experiment:
    context = _build_soma_tiledb_context(aws_region).replace(
        tiledb_config={
            "py.init_buffer_bytes": buffer_bytes,
            "soma.init_buffer_bytes": buffer_bytes,
        }
    )
    return soma.Experiment.open(uri, context=context)


class _ObsAndXIterator(Iterator[ObsDatum]):
    """
    Iterates through a set of obs and related X rows, specified as `soma_joinid`s. Encapsulates the batch-based data
    fetching from TileDB-SOMA objects, providing row-based iteration.
    """

    obs_tables_iter: somacore.ReadIter[pa.Table]
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
        exp_uri: str,
        aws_region: Optional[str],
        measurement_name: str,
        X_layer_name: str,
        batch_size: int,
        encoders: Dict[str, LabelEncoder],
        stats: Stats,
        obs_column_names: Sequence[str],
        dense_X: bool,
        buffer_bytes: int = DEFAULT_BUFFER_BYTES,
    ) -> None:
        self.obs_joinids = obs_joinids
        self.var_joinids = var_joinids
        self.batch_size = batch_size
        self.dense_X = dense_X

        # holding reference to SOMA object prevents it from being closed
        self.exp = _open_experiment(exp_uri, aws_region, buffer_bytes=buffer_bytes)
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
            return torch.Tensor(X.todense()), obs_tensor
        else:
            coo = X.tocoo()

            X_tensor = torch.sparse_coo_tensor(
                # Note: The `np.array` seems unnecessary, but PyTorch warns bare array is "extremely slow"
                indices=torch.Tensor(np.array([coo.row, coo.col])),
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

        pytorch_logger.debug("Retrieving next TileDB-SOMA batch...")
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
        self.stats.elapsed += int(time() - start_time)
        self.stats.n_soma_batches += 1
        pytorch_logger.debug(f"Retrieved batch: shape={self.X_batch.shape}, cum_stats: {self.stats}")
        return self.obs_batch_


class ExperimentDataPipe(pipes.IterDataPipe[Dataset[ObsDatum]]):  # type: ignore
    """
    An iterable-style data pipe.
    """

    _query: Optional[soma.ExperimentAxisQuery]

    _obs_joinids_partitioned: Optional[pa.Array]

    _obs_and_x_iter: Optional[_ObsAndXIterator]

    _encoders: Optional[Encoders]

    _stats: Stats

    def __init__(
        self,
        experiment: soma.Experiment,
        ms_name: str,
        layer_name: str,
        obs_query: Optional[soma.AxisQuery] = None,
        var_query: Optional[soma.AxisQuery] = None,
        obs_column_names: Sequence[str] = (),
        batch_size: int = 1,
        dense_X: bool = False,
        num_workers: int = 0,
        buffer_bytes: int = DEFAULT_BUFFER_BYTES,
    ) -> None:
        if num_workers > 1 and not dense_X:
            raise NotImplementedError(
                "torch does not work with sparse tensors in multi-processing mode "
                "(see https://github.com/pytorch/pytorch/issues/20248)"
            )

        self.exp_uri = experiment.uri
        self.aws_region = experiment.context.tiledb_ctx.config().get("vfs.s3.region")
        self.ms_name = ms_name
        self.layer_name = layer_name
        self.obs_query = obs_query
        self.var_query = var_query
        self.obs_column_names = obs_column_names
        self.batch_size = batch_size
        self.dense_X = dense_X
        # TODO: This will control the obs SOMA batch sizes, and should be replaced with a row count once TileDB-SOMA
        #  supports `batch_size` param. Unfortunately, the buffer_bytes also impacts the X data fetch efficiency, which
        #  should be tuned independently.
        self.buffer_bytes = buffer_bytes
        self._query = None
        self._stats = Stats()
        self._encoders = None

        if "soma_joinid" not in self.obs_column_names:
            self.obs_column_names = ["soma_joinid", *self.obs_column_names]

    def _init(self) -> None:
        if self._query is not None:
            return

        # TODO: support multiple layers, per somacore.query.query.ExperimentAxisQuery.to_anndata()
        # TODO: for dev-time use, specify smaller batch size via platform_config, once supported

        # TODO: handle closing of exp when iterator is no longer in use; may need be used as a ContextManager,
        #  but not clear how we can do that when used by DataLoader
        exp = _open_experiment(self.exp_uri, self.aws_region, buffer_bytes=self.buffer_bytes)

        self._query = exp.axis_query(
            measurement_name=self.ms_name,
            obs_query=self.obs_query,
            var_query=self.var_query,
        )

        obs_joinids = self._query.obs_joinids()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            partition, num_partitions = worker_info.id, worker_info.num_workers
        else:
            partition, num_partitions = 0, 1

        if num_partitions is not None:
            # partitioned data loading
            # NOTE: Can alternately use a `worker_init_fn` to split among workers split workload
            partition_size = len(obs_joinids) // num_partitions
            partition_start = partition_size * partition
            partition_end_excl = min(len(obs_joinids), partition_start + partition_size)
            self._obs_joinids_partitioned = obs_joinids[partition_start:partition_end_excl]

            pytorch_logger.debug(
                f"Process {os.getpid()} handling partition {partition + 1} of {num_partitions}, "
                f"range={partition_start}:{partition_end_excl}, "
                f"{partition_size:}"
            )

    def __iter__(self) -> Iterator[ObsDatum]:
        self._init()
        assert self._query is not None

        return _ObsAndXIterator(
            obs_joinids=self._obs_joinids_partitioned or self._query.obs_joinids(),
            var_joinids=self._query.var_joinids(),
            exp_uri=self.exp_uri,
            aws_region=self.aws_region,
            measurement_name=self.ms_name,
            X_layer_name=self.layer_name,
            batch_size=self.batch_size,
            encoders=self.obs_encoders(),
            stats=self._stats,
            obs_column_names=self.obs_column_names,
            dense_X=self.dense_X,
            buffer_bytes=self.buffer_bytes,
        )

    def __len__(self) -> int:
        self._init()
        assert self._query is not None

        return int(self._query.n_obs)

    def __getitem__(self, index: int) -> ObsDatum:
        raise NotImplementedError("IterDataPipe can only be iterated")

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # Don't pickle `_query`
        del state["_query"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._query = None

    def obs_encoders(self) -> Encoders:
        if self._encoders is not None:
            return self._encoders

        self._init()
        assert self._query is not None

        obs = self._query.obs(column_names=self.obs_column_names).concat()
        self._encoders = {}
        for col in self.obs_column_names:
            if obs[col].type in (pa.string(), pa.large_string()):
                enc = LabelEncoder()
                enc.fit(obs[col].combine_chunks().unique())
                self._encoders[col] = enc

        return self._encoders

    def stats(self) -> Stats:
        return self._stats

    @property
    def shape(self) -> Tuple[int, int]:
        self._init()
        assert self._query is not None

        return self._query.n_obs, self._query.n_vars


# Note: must be a top-level function (and not a lambda), to play nice with multiprocessing pickling
def collate_noop(x: Any) -> Any:
    return x


# TODO: Move into somacore.ExperimentAxisQuery
def experiment_dataloader(
    datapipe: pipes.IterDataPipe,
    num_workers: int = 0,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """
    Factory method for PyTorch DataLoader. Provides a safer, more convenient interface for instantiating a DataLoader
    that works with the ExperimentDataPipe, since not all of DataLoader's params can be used (batch_size, sampler,
    batch_sampler, collate_fn).
    """

    unsupported_dataloader_args = ["sampler", "batch_sampler", "collate_fn"]
    if set(unsupported_dataloader_args).intersection(dataloader_kwargs.keys()):
        raise ValueError(f"The {','.join(unsupported_dataloader_args)} DataLoader params are not supported")

    if "batch_size" in dataloader_kwargs:
        del dataloader_kwargs["batch_size"]

    return DataLoader(
        datapipe,
        batch_size=None,  # batching is handled by our ExperimentDataPipe
        num_workers=num_workers,
        # avoid use of default collator, which adds an extra (3rd) dimension to the tensor batches
        collate_fn=collate_noop,
        **dataloader_kwargs,
    )


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
        dense_X_arg,
        torch_batch_size_arg,
        num_workers_arg,
    ) = sys.argv[1:]

    census = cellxgene_census.open_soma(uri=census_uri_arg)

    exp_datapipe = ExperimentDataPipe(
        experiment=census["census_data"]["homo_sapiens"],
        ms_name=measurement_name_arg,
        layer_name=layer_name_arg,
        obs_query=soma.AxisQuery(value_filter=(obs_value_filter_arg or None)),
        var_query=soma.AxisQuery(coords=(slice(1, 9),)),
        obs_column_names=column_names_arg.split(","),
        batch_size=int(torch_batch_size_arg),
        num_workers=int(num_workers_arg),
        buffer_bytes=2**12,
        dense_X=dense_X_arg.lower() == "dense",
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
