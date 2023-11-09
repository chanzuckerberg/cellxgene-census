from __future__ import annotations

from abc import ABCMeta, abstractproperty
from contextlib import AbstractContextManager
from typing import Any, Iterator, Tuple, Union, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute
import tiledbsoma as soma
from typing_extensions import Self

from .census_util import get_obs_soma_joinids
from .metadata import EmbeddingMetadata
from .util import get_logger, soma_context

logger = get_logger()


EmbeddingTableIterator = Iterator[pa.Table]


class EmbeddingIJDPipe(EmbeddingTableIterator, AbstractContextManager["EmbeddingIJDPipe"], metaclass=ABCMeta):
    """
    Returns pa.Table with I, J and D columns (i.e., COO), in row-major/C sorted order.
    Must not have dups.
    """

    def __iter__(self) -> EmbeddingTableIterator:
        return self

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        ...

    @abstractproperty
    def type(self) -> pa.DataType:
        ...

    @abstractproperty
    def value_range(self) -> Union[Tuple[float, float], Tuple[None, None]]:
        """Return [min, max]"""
        ...


class SOMAIJDPipe(EmbeddingIJDPipe):
    def __init__(self, uri: str):
        self.uri = uri

    def __enter__(self) -> Self:
        self._A: soma.SparseNDArray = soma.open(self.uri, context=soma_context())
        if self._A.soma_type != "SOMASparseNDArray" or self._A.ndim != 2:
            raise ValueError("Must be a 2D SOMA SparseNDArray")
        self._reader = (
            tbl.rename_columns(["i", "j", "d"])
            for tbl, _ in self._A.read(result_order="row-major")
            .blockwise(axis=0, size=2**20, reindex_disable_on_axis=[0, 1])
            .tables()
        )
        return self

    def __exit__(self, *_: Any) -> None:
        self._reader.close()
        self._A.close()

    def __next__(self) -> pa.Table:
        return next(self._reader)

    @property
    def shape(self) -> Tuple[int, ...]:
        return cast(Tuple[int, ...], self._A.shape)

    @property
    def type(self) -> pa.DataType:
        return self._A.schema.field("soma_data").type

    @property
    def value_range(self) -> Union[Tuple[float, float], Tuple[None, None]]:
        logger.debug("SOMAIJDPipe - scanning for value_range")
        vrng: Union[Tuple[float, float], Tuple[None, None]] = (None, None)

        with soma.open(self.uri, context=soma_context()) as A:
            if A.nnz > 0:
                for tbl in A.read().tables():
                    min_max = pa.compute.min_max(tbl.column("soma_data"))
                    _min, _max = min_max["min"].as_py(), min_max["max"].as_py()
                    vrng = (
                        _min if vrng[0] is None else min(_min, vrng[0]),
                        _max if vrng[1] is None else min(_max, vrng[1]),
                    )

        logger.debug(f"SOMAIJDPipe - value_range is {vrng}")
        return vrng


class TestDataIJDPipe(EmbeddingIJDPipe):
    def __init__(self, n_obs: int, n_features: int, metadata: EmbeddingMetadata):
        rng = np.random.default_rng()
        self._scale = 2.0
        self._offset = -0.1

        all_obs, obs_shape = get_obs_soma_joinids(metadata)
        if not n_obs:
            n_obs = len(all_obs)
        if n_obs == len(all_obs):
            obs_joinids = all_obs
        else:
            obs_joinids = rng.choice(all_obs, n_obs, replace=False)
        obs_joinids = np.sort(obs_joinids)

        self.n_obs = n_obs
        self.n_features = n_features
        self.obs_joinids = obs_joinids
        self.rng = rng

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
    def shape(self) -> Tuple[int, ...]:
        return (self.n_obs, self.n_features)

    @property
    def type(self) -> pa.DataType:
        return pa.float32()

    @property
    def value_range(self) -> Union[Tuple[float, float], Tuple[None, None]]:
        if self.n_obs == 0:
            return (None, None)
        return (self._offset, self._scale + self._offset)


def test_embedding(n_obs: int, n_features: int, metadata: EmbeddingMetadata) -> EmbeddingIJDPipe:
    return TestDataIJDPipe(n_obs, n_features, metadata)


def soma_ingest(soma_uri: str, _: EmbeddingMetadata) -> EmbeddingIJDPipe:
    return SOMAIJDPipe(soma_uri)


def npy_ingest(joinid_uri: str, embedding_uri: str, metadata: EmbeddingMetadata) -> EmbeddingIJDPipe:
    raise NotImplementedError()


def csv_ingest(csv_uri: str, metadata: EmbeddingMetadata) -> EmbeddingIJDPipe:
    # only partially implemented
    raise NotImplementedError()

    # SAVE for now

    # def skip_comment(row):
    #     if row.text.startswith("# "):
    #         return "skip"
    #     else:
    #         return "error"

    # parse_opts = {"invalid_row_handler": skip_comment}
    # if args.csv_uri.endswith(".csv"):
    #     parse_opts["delimiter"] = ","
    # elif args.csv_uri.endswith(".tsv"):
    #     parse_opts["delimiter"] = "\t"
    # parse_options = csv.ParseOptions(*parse_opts)

    # tbl = csv.read_csv(
    #     args.csv_uri, parse_options=parse_options, invalid_row_handler=skip_comment
    # )

    # # Expect column names:
    # #  soma_joinid
    # #  0..N

    # # first drop unexpected columns - i.e., by name or by type
    # drop = []
    # for n in range(tbl.num_columns):
    #     field = tbl.schema.field(n)
    #     if field.name == "soma_joinid":
    #         if not pa.types.is_integer(field.type):
    #             drop.append(field.name)
    #         continue

    #     if not pa.types.is_floating(field.type):
    #         drop.append(field.name)
    #         continue

    #     try:
    #         int(field.name)
    #     except ValueError:
    #         drop.append(field.name)

    # if drop:
    #     tbl = tbl.drop_columns(drop)
