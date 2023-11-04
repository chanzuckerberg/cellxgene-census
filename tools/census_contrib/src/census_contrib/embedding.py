from __future__ import annotations

from typing import Any, Tuple, TYPE_CHECKING, cast

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa

import cellxgene_census
import tiledbsoma as soma

from .metadata import ContribMetadata
from .util import error

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
    # Get joinids
    obs_soma_joinids = get_obs_soma_joinids(metadata)

    # Load embedding
    with soma.open(args.soma_uri) as A:
        emb_sparse_tbl = A.read().tables().concat()

    raise NotImplementedError()


def csv_ingest(args: "Arguments", metadata: ContribMetadata) -> EmbeddingIJD:
    raise NotImplementedError()


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

    i = np.empty((n_obs, n_features), dtype=np.int64)
    i.T[:] = obs_joinids
    i = i.ravel()

    j = np.empty((n_obs, n_features), dtype=np.int64)
    j[:] = np.arange(n_features)
    j = j.ravel()

    d = rng.random((n_obs * n_features), dtype=np.float32)

    emb = EmbeddingIJD(i=i, j=j, d=d, shape=(obs_shape[0], n_features))
    validate_embedding(args, metadata, emb)
    return emb


def get_obs_soma_joinids(
    metadata: ContribMetadata,
) -> Tuple[npt.NDArray[np.int64], Tuple[int, ...]]:
    """Return experiment obs soma_joind values and obs shape"""
    with cellxgene_census.open_soma(census_version=metadata.census_version) as census:
        exp = census["census_data"][metadata.experiment_name]
        tbl = exp.obs.read(column_names=["soma_joinid"]).concat()

        joinids = cast(npt.NDArray[np.int64], tbl.column("soma_joinid").to_numpy())
        return joinids, (joinids.max() + 1,)


def validate_embedding(
    args: Arguments, metadata: ContribMetadata, emb: pa.Table
) -> None:
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

    # Verify I values all exist as legit soma_joinids
    if not np.isin(emb.i, obs_joinids).all():
        error(args, "Embedding contains joinids not present in experiment obs")

    # Verify all J values are in expected domain
    if np.logical_or((emb.j < 0), (emb.j > metadata.n_features)).any():
        error(args, "Embedding J values not in range [0, n_features)")

    # Must have at least one cell embedded
    if len(emb.i) < 1:
        error(args, "Embedding must have at least one cell embedded")
