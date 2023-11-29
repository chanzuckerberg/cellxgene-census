# Copyright (c) 2022-2023 Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.

"""
Methods to support simplifed access to community contributed embeddings.
"""
from __future__ import annotations

import json
import warnings
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import tiledbsoma as soma

from .._open import _build_soma_tiledb_context, open_soma
from .._release_directory import get_census_version_directory


def get_embedding(
    census_version: str,
    embedding_uri: str,
    obs_soma_joinids: Union[npt.NDArray[np.int64], pa.Array],
    context: Optional[soma.options.SOMATileDBContext] = None,
) -> npt.NDArray[np.float32]:
    if isinstance(obs_soma_joinids, (pa.Array, pa.ChunkedArray)):
        obs_soma_joinids = obs_soma_joinids.to_numpy()
    if obs_soma_joinids.dtype != np.int64:
        raise TypeError("obs_soma_joinids must be array of int64")

    # Currently, all embeddings are hosted in us-west-2 so use that as a default.
    # Allow the user to override for exceptional cases.
    context = _build_soma_tiledb_context("us-west-2", context)

    # Attempt to resolve census version aliases
    census_directory = get_census_version_directory()
    resolved_census_version = census_directory.get(census_version, None)

    with soma.open(embedding_uri, context=context) as E:
        embedding_metadata = json.loads(E.metadata["CxG_contrib_metadata"])

        if resolved_census_version is None:
            warnings.warn(
                "Unable to determine Census version - skipping validation of Census and embedding version.", stacklevel=1
            )
        elif resolved_census_version != census_directory.get(embedding_metadata["census_version"], None):
            raise ValueError("Census and embedding mismatch - census_version not equal")

        with open_soma(census_version=census_version, context=context) as census:
            experiment_name = embedding_metadata["experiment_name"]
            if experiment_name not in census["census_data"]:
                raise ValueError("Census and embedding mismatch - experiment_name does not exist")
            measurement_name = embedding_metadata["measurement_name"]
            if measurement_name not in census["census_data"][experiment_name].ms:
                raise ValueError("Census and embedding mismatch - measurement_name does not exist")

        embedding_shape = (len(obs_soma_joinids), E.shape[1])
        embedding = np.full(embedding_shape, np.NaN, dtype=np.float32, order="C")

        obs_indexer = pd.Index(obs_soma_joinids)
        for tbl in E.read(coords=(obs_soma_joinids,)).tables():
            obs_idx = obs_indexer.get_indexer(tbl.column("soma_dim_0").to_numpy())  # type: ignore[no-untyped-call]
            feat_idx = tbl.column("soma_dim_1").to_numpy()
            emb = tbl.column("soma_data")

            indices = obs_idx * E.shape[1] + feat_idx
            np.put(embedding.reshape(-1), indices, emb)

    return embedding
