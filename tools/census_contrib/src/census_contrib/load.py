from __future__ import annotations

from abc import ABCMeta, abstractproperty
from contextlib import AbstractContextManager
from typing import Any, Iterator, Tuple, cast

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import tiledbsoma as soma
from typing_extensions import Self

from .census_util import get_obs_soma_joinids
from .metadata import EmbeddingMetadata
from .util import get_logger, soma_context

logger = get_logger()


def is_ndarray(emb: EmbeddingIJD, attr: attrs.Attribute[Any], value: Any) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{attr.name} - must be a numpy.ndarray")


def is_npt_int64(emb: EmbeddingIJD, attr: attrs.Attribute[Any], value: Any) -> None:
    is_ndarray(emb, attr, value)
    if not value.dtype == np.int64:
        raise TypeError(f"{attr.name} - must have dtype int64")


def is_npt_float32(emb: EmbeddingIJD, attr: attrs.Attribute[Any], value: Any) -> None:
    is_ndarray(emb, attr, value)
    if not value.dtype == np.float32:
        raise TypeError(f"{attr.name} - must have dtype float32")


@attrs.define(kw_only=True, frozen=True)
class EmbeddingIJD:
    # TODO: it would be more efficient if this stored Arrow ChunkedArray rather than NDArray.

    i: npt.NDArray[np.int64] = attrs.field(
        validator=[is_npt_int64, attrs.validators.min_len(1)],
    )
    j: npt.NDArray[np.int64] = attrs.field(
        validator=[is_npt_int64, attrs.validators.min_len(1)],
    )
    d: npt.NDArray[np.float32] = attrs.field(
        validator=[is_npt_float32, attrs.validators.min_len(1)],
    )
    shape: Tuple[int, int]


def npy_ingest(joinid_uri: str, embedding_uri: str, metadata: EmbeddingMetadata) -> EmbeddingIJD:
    raise NotImplementedError()


def soma_ingest(soma_uri: str, metadata: EmbeddingMetadata) -> EmbeddingIJD:
    # Load embedding
    logger.info(f"Loading {soma_uri}")
    with soma.open(soma_uri, context=soma_context()) as A:
        shape = A.shape
        emb_tbl = A.read(result_order="row-major").tables().concat()

    logger.info(f"Flattening {soma_uri}")
    i = emb_tbl.column("soma_dim_0").to_numpy()
    j = emb_tbl.column("soma_dim_1").to_numpy()
    d = emb_tbl.column("soma_data").to_numpy()
    emb = EmbeddingIJD(i=i, j=j, d=d, shape=shape)

    logger.info(f"Finished loadding {soma_uri}")
    return emb


def csv_ingest(csv_uri: str, metadata: EmbeddingMetadata) -> EmbeddingIJD:
    # only partially implemented
    raise NotImplementedError()

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


def test_embedding(n_obs: int, n_features: int, metadata: EmbeddingMetadata) -> EmbeddingIJD:
    """Generate a test embedding containing random data."""
    rng = np.random.default_rng()

    n_features = n_features
    n_obs = n_obs  # if zero or None, embed every obs. Else, random selection

    all_obs, obs_shape = get_obs_soma_joinids(metadata)
    if not n_obs:
        n_obs = len(all_obs)
    if n_obs == len(all_obs):
        obs_joinids = all_obs
    else:
        obs_joinids = rng.choice(all_obs, n_obs, replace=False)
    obs_joinids = np.sort(obs_joinids)

    i = np.empty((n_obs, n_features), dtype=np.int64)
    i.T[:] = obs_joinids
    i = i.ravel()

    j = np.empty((n_obs, n_features), dtype=np.int64)
    j[:] = np.arange(n_features)
    j = j.ravel()

    d = 2.0 * rng.random((n_obs * n_features), dtype=np.float32) - 0.1

    return EmbeddingIJD(i=i, j=j, d=d, shape=(obs_shape[0], n_features))


#
#
#

EmbeddingTableIterator = Iterator[pa.Table]


class EmbeddingIJDPipe(EmbeddingTableIterator, AbstractContextManager["EmbeddingIJDPipe"], metaclass=ABCMeta):
    """
    Returns pa.Table with I, J and D columns (i.e., COO), in row-major/C sorted order.
    Must not have dups.
    """

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        ...

    @abstractproperty
    def type(self) -> pa.DataType:
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

    def __iter__(self) -> EmbeddingTableIterator:
        return self

    def __next__(self) -> pa.Table:
        return next(self._reader)

    @property
    def shape(self) -> Tuple[int, ...]:
        return cast(Tuple[int, ...], self._A.shape)

    @property
    def type(self) -> pa.DataType:
        return self._A.schema.field("soma_data").type


class TestDataIJDPipe(EmbeddingIJDPipe):
    def __init__(self, n_obs: int, n_features: int, metadata: EmbeddingMetadata):
        rng = np.random.default_rng()

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

    def __iter__(self) -> EmbeddingTableIterator:
        return self

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

        d = 2.0 * self.rng.random((n_obs * self.n_features), dtype=np.float32) - 0.1

        return pa.Table.from_pydict({"i": i, "j": j, "d": d})

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.n_obs, self.n_features)

    @property
    def type(self) -> pa.DataType:
        return pa.float32()
