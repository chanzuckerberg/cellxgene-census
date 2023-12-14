from __future__ import annotations

from abc import ABCMeta, abstractproperty
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import tiledbsoma as soma
from typing_extensions import Self

from .census_util import get_obs_soma_joinids
from .config import Config
from .util import blocksize, blockwise_axis0_tables, get_logger, has_blockwise_iterator, soma_context

logger = get_logger()


EmbeddingTableIterator = Iterator[pa.Table]
EmbeddingIJDDomains = Dict[Literal["i", "j", "d"], Union[Tuple[float, float], Tuple[None, None]]]


class EmbeddingIJDPipe(EmbeddingTableIterator, AbstractContextManager["EmbeddingIJDPipe"], metaclass=ABCMeta):
    """
    Returns pa.Table with i, j, and d columns (i.e., COO), in row-major/C sorted order.
    Must not have dups.
    """

    def __iter__(self) -> EmbeddingTableIterator:
        return self

    @abstractproperty
    def type(self) -> pa.DataType:
        ...

    @abstractproperty
    def domains(self) -> EmbeddingIJDDomains:
        """Return domains of i, j, and d"""


class SOMAIJDPipe(EmbeddingIJDPipe):
    def __init__(self, uri: Path):
        self.uri = uri

    def __enter__(self) -> Self:
        self._A: soma.SparseNDArray = soma.open(self.uri.as_posix(), context=soma_context())
        if self._A.soma_type != "SOMASparseNDArray" or self._A.ndim != 2:
            raise ValueError("Must be a 2D SOMA SparseNDArray")
        size = blocksize(self._A.shape[1])

        self._reader = (
            (
                tbl.rename_columns(["i", "j", "d"])
                for tbl, _ in self._A.read(result_order="row-major")
                .blockwise(axis=0, size=size, reindex_disable_on_axis=[0, 1])
                .tables()
            )
            if has_blockwise_iterator()
            else (
                tbl.rename_columns(["i", "j", "d"])
                for tbl, _ in blockwise_axis0_tables(
                    self._A, result_order="row-major", size=size, reindex_disable_on_axis=[0, 1]
                )
            )
        )
        return self

    def __exit__(self, *_: Any) -> None:
        self._reader.close()
        self._A.close()

    def __next__(self) -> pa.Table:
        return next(self._reader)

    @property
    def type(self) -> pa.DataType:
        return self._A.schema.field("soma_data").type

    @property
    def domains(self) -> EmbeddingIJDDomains:
        """Return the domains of i, j and d"""
        logger.debug("SOMAIJDPipe - scanning for domains")

        _domains: EmbeddingIJDDomains = {"i": (None, None), "j": (None, None), "d": (None, None)}

        def accum_min_max(tbl: pa.Table, col_name: str, col_alias: Literal["i", "j", "d"]) -> None:
            min_max = pa.compute.min_max(tbl.column(col_name))
            _min, _max = min_max["min"].as_py(), min_max["max"].as_py()
            domain = _domains[col_alias]
            domain = (
                _min if domain[0] is None else min(_min, domain[0]),
                _max if domain[1] is None else max(_max, domain[1]),
            )
            _domains[col_alias] = domain

        with soma.open(self.uri.as_posix(), context=soma_context()) as A:
            if A.nnz > 0:
                for tbl in A.read().tables():
                    accum_min_max(tbl, "soma_data", "d")
                    accum_min_max(tbl, "soma_dim_0", "i")
                    accum_min_max(tbl, "soma_dim_1", "j")

        logger.debug(f"SOMAIJDPipe - found domains {_domains}")

        return _domains


class NPYIJDPipe(EmbeddingIJDPipe):
    """
    Basic approach:
    1. load joinid 1d array as npy or txt
    2. argsort joinid array as there is no requirement it is 0..n
    3. mmap emb array r/o
    4. yield chunks
    """

    def __init__(self, joinids_uri: Path, embeddings_uri: Path):
        self.joinids_uri = joinids_uri
        self.embeddings_uri = embeddings_uri

    def __enter__(self) -> Self:
        logger.info("NPYIJDPipe - loading")
        self.joinids = self._load_joinids()
        self.joinids_sort = np.argsort(self.joinids)
        self.embeddings = np.load(self.embeddings_uri, mmap_mode="r")

        self.n_obs = len(self.joinids)
        self.n_features = self.embeddings.shape[1]
        if self.n_obs != self.embeddings.shape[0]:
            raise ValueError("Embedding NPY and joinid files do not have compatible shape")

        if self.embeddings.dtype != np.float32:
            raise ValueError("Embedding NPY must be float32")

        # step through n_obs in blocks
        self.step_size = blocksize(self.n_features)
        self._steps = (i for i in range(0, self.n_obs, self.step_size))

        logger.info(f"NPYIJDPipe - found n_obs={self.n_obs}, embeddings shape {self.embeddings.shape}")
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def __next__(self) -> pa.Table:
        next_step = next(self._steps)
        pnts = self.joinids_sort[next_step : next_step + self.step_size]
        n_obs = len(pnts)

        i = np.empty((n_obs, self.n_features), dtype=np.int64)
        i.T[:] = self.joinids[pnts]
        i = i.ravel()

        j = np.empty((n_obs, self.n_features), dtype=np.int64)
        j[:] = np.arange(self.n_features)
        j = j.ravel()

        d = self.embeddings[pnts, :].ravel()

        return pa.Table.from_pydict({"i": i, "j": j, "d": d})

    @property
    def type(self) -> pa.DataType:
        return pa.from_numpy_dtype(self.embeddings.dtype)

    @property
    def domains(self) -> EmbeddingIJDDomains:
        """Return the domains of i, j and d"""
        logger.debug("NPYIJDPipe - scanning for domains")

        min_max = pa.compute.min_max(pa.array(self.embeddings.ravel()))
        _min, _max = min_max["min"].as_py(), min_max["max"].as_py()
        _domains: EmbeddingIJDDomains = {
            "i": (self.joinids[self.joinids_sort[0]], self.joinids[self.joinids_sort[-1]]),
            "j": (0, self.n_features - 1),
            "d": (_min, _max),
        }
        logger.debug(f"NPYIJDPipe - found domains {_domains}")
        return _domains

    def _load_joinids(self) -> npt.NDArray[np.int64]:
        logger.info(f"Loading joinids from {self.joinids_uri}")

        if self.joinids_uri.suffix == ".txt":
            joinids = np.loadtxt(self.joinids_uri, dtype=np.int64)
        elif self.joinids_uri.suffix == ".npy":
            joinids = np.load(self.joinids_uri)
        else:
            raise ValueError("joinid file has unrecoginized format (extension) - expect .txt or .npy")

        if joinids.dtype != np.int64:
            raise ValueError("Joinids are not int64")

        logger.info(f"Loaded {len(joinids)} joinids")
        return joinids


class TestDataIJDPipe(EmbeddingIJDPipe):
    def __init__(self, n_obs: int, n_features: int, config: Config):
        rng = np.random.default_rng(seed=0)
        self._scale = 2.0
        self._offset = -0.1

        all_obs, obs_shape = get_obs_soma_joinids(config)
        if n_obs == len(all_obs):
            obs_joinids = all_obs
        else:
            obs_joinids = rng.choice(all_obs, n_obs, replace=False)
        obs_joinids = np.sort(obs_joinids)

        self.n_obs = n_obs
        self.n_features = n_features
        self.obs_joinids = obs_joinids
        self.rng = rng
        self.obs_shape = obs_shape

        # step through n_obs in blocks
        self.step_size = 2**20
        self._steps = (i for i in range(0, len(obs_joinids), self.step_size))

    def __exit__(self, *_: Any) -> None:
        pass

    def __next__(self) -> pa.Table:
        next_step = next(self._steps)
        next_block = self.obs_joinids[next_step : next_step + self.step_size]
        n_obs = len(next_block)

        i = np.empty((n_obs, self.n_features), dtype=np.int64)
        i.T[:] = next_block
        i = i.ravel()

        j = np.empty((n_obs, self.n_features), dtype=np.int64)
        j[:] = np.arange(self.n_features)
        j = j.ravel()

        d = self._scale * self.rng.random((n_obs * self.n_features), dtype=np.float32) + self._offset

        return pa.Table.from_pydict({"i": i, "j": j, "d": d})

    @property
    def type(self) -> pa.DataType:
        return pa.float32()

    @property
    def domains(self) -> EmbeddingIJDDomains:
        if self.n_obs == 0:
            return {"i": (None, None), "j": (None, None), "d": (None, None)}

        return {
            "i": (0, self.obs_shape[0] - 1),
            "j": (0, self.n_features - 1),
            "d": (self._offset, self._scale + self._offset),
        }


def test_embedding(n_obs: int, n_features: int, config: Config) -> EmbeddingIJDPipe:
    return TestDataIJDPipe(n_obs, n_features, config)


def soma_ingest(soma_path: Path, _: Config) -> EmbeddingIJDPipe:
    return SOMAIJDPipe(soma_path)


def npy_ingest(joinid_path: Path, embedding_path: Path, _: Config) -> EmbeddingIJDPipe:
    return NPYIJDPipe(joinid_path, embedding_path)
