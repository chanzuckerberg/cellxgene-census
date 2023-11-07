from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, TypeVar, cast

import attrs
import cellxgene_census
import numba as nb
import numpy as np
import numpy.typing as npt
import tiledbsoma as soma

from .metadata import ContribMetadata
from .util import error, soma_context

if TYPE_CHECKING:
    from .args import Arguments


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


def npy_ingest(args: "Arguments", metadata: ContribMetadata) -> EmbeddingIJD:
    raise NotImplementedError()


def soma_ingest(args: "Arguments", metadata: ContribMetadata) -> EmbeddingIJD:
    # Load embedding
    args.logger.info(f"Loading {args.soma_uri}")
    with soma.open(args.soma_uri, context=soma_context()) as A:
        shape = A.shape
        emb_tbl = A.read(result_order="row-major").tables().concat()

    args.logger.info(f"Flattening {args.soma_uri}")
    i = emb_tbl.column("soma_dim_0").to_numpy()
    j = emb_tbl.column("soma_dim_1").to_numpy()
    d = emb_tbl.column("soma_data").to_numpy()
    emb = EmbeddingIJD(i=i, j=j, d=d, shape=shape)

    args.logger.info(f"Finished loadding {args.soma_uri}")
    return emb


def csv_ingest(args: "Arguments", metadata: ContribMetadata) -> EmbeddingIJD:
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


def test_embedding(args: "Arguments", metadata: ContribMetadata) -> EmbeddingIJD:
    """Generate a test embedding."""
    rng = np.random.default_rng()

    n_features = args.n_features
    n_obs = args.n_obs  # if zero or None, embed every obs. Else, random selection

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


def get_obs_soma_joinids(
    metadata: ContribMetadata,
) -> Tuple[npt.NDArray[np.int64], Tuple[int, ...]]:
    """Return experiment obs soma_joind values and obs shape"""
    with cellxgene_census.open_soma(census_version=metadata.census_version) as census:
        exp = census["census_data"][metadata.experiment_name]
        tbl = exp.obs.read(column_names=["soma_joinid"]).concat()

        joinids = cast(npt.NDArray[np.int64], tbl.column("soma_joinid").to_numpy())
        return joinids, (joinids.max() + 1,)


def validate_embedding(args: Arguments, metadata: ContribMetadata, emb: EmbeddingIJD) -> None:
    """
    Will error/exit on invalid embedding.

    1. Embedding shape must be (O, M), where O is the domain of the associated Census experiment
    obs dataframe, and M is user defined (e.g., for a 2D UMAP, M would be 2).
    2. All dim0 values in the embedding must have corresponding obs soma_joinid in the corresponding Census experiment
    3. An embedding must have at least one (1) cell embedded
    4. Embedding type must be float32

    """

    obs_joinids, obs_shape = get_obs_soma_joinids(metadata)

    # Verify types
    if emb.i.dtype != np.int64 or emb.j.dtype != np.int64 or emb.d.dtype != np.float32:
        error(args, "Embedding data types not int64/int64/float32")

    # Embedding shape
    if len(emb.shape) != 2:
        error(args, "Embedding must be 2D")
    if emb.shape[0] != obs_shape[0]:
        error(args, "Embedding and obs shape differ.")
    if emb.shape[1] != metadata.n_features:
        error(
            args,
            "Embedding and metadata specify a different number of embedding features.",
        )
    if emb.i.shape != emb.j.shape or emb.i.shape != emb.d.shape:
        error(args, "Malformed embedding COO")

    # Must have at least one cell embedded
    if len(emb.i) < 1:
        error(args, "Embedding must have at least one cell embedded")

    # Verify I values all exist as legit soma_joinids
    if not isin_all(emb.i, obs_joinids):
        error(args, "Embedding contains joinids not present in experiment obs")

    # Verify all J values are in expected domain
    if not is_in_range_all(emb.j, 0, metadata.n_features - 1):
        error(args, "Embedding J values not in range [0, n_features)")

    # Embedding must be sorted with no dups (test assumes feature indices [0, N))
    if not is_sorted_unique(emb.i, emb.j, emb.shape[1]):
        error(args, "Embedding must be sorted with no duplicate coordinates")


_NPT = TypeVar("_NPT", bound=npt.NDArray[np.number[Any]])


@nb.njit()  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def isin_all(elmts: _NPT, test_elmts: _NPT) -> bool:
    """
    Return equivalent of numpy.isin(elmts, test_elmts).all() without the
    memory allocation and extra reduction required by the numpy expression.
    """
    test = set(test_elmts)
    for i in range(len(elmts)):
        if elmts[i] not in test:
            return False
    return True


@nb.njit()  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def is_in_range_all(elmts: _NPT, min: float, max: float) -> bool:
    """
    Return equivalent of np.logical_or((elmts < min), (elmts > max)).any()
    without the memory allocation and extra reduction required by the numpy expression.
    """
    for i in range(len(elmts)):
        if elmts[i] < min or elmts[i] > max:
            print(elmts[i], min, max)
            return False
    return True


@nb.njit()  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def is_sorted_unique(i: npt.NDArray[np.int64], j: npt.NDArray[np.int64], j_shape: int) -> bool:
    last_coord = -1
    for n in range(len(i)):
        c_coord = i[n] * j_shape + j[n]
        if c_coord <= last_coord:
            return False
        last_coord = c_coord
    return True
