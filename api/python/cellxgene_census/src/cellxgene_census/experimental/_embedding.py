# Copyright (c) 2022, Chan Zuckerberg Initiative
#
# Licensed under the MIT License.

"""Methods to support simplifed access to community contributed embeddings."""
from __future__ import annotations

import json
import warnings
from typing import Any, Dict, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import tiledbsoma as soma

from .._open import get_default_soma_context, open_soma
from .._release_directory import get_census_version_directory


def get_embedding_metadata(embedding_uri: str, context: soma.options.SOMATileDBContext | None = None) -> dict[str, Any]:
    """Read embedding metadata and return as a Python dict.

    Args:
        embedding_uri:
            The embedding URI.
        context:
            A custom :class:`tiledbsoma.SOMATileDBContext` which will be used to open the SOMA object. Optional,
            defaults to ``None``.

    Returns:
        A Python dictionary containing metadata describing the embedding.

    Examples:
        >>> get_experiment_metadata(uri)

    """
    # Allow the user to override context for exceptional cases (e.g. the aws region)
    context = context or get_default_soma_context()

    with soma.open(embedding_uri, context=context) as E:
        # read embedding metadata and decode the JSON-encoded string
        embedding_metadata = json.loads(E.metadata["CxG_embedding_info"])
        assert isinstance(embedding_metadata, dict)

    return cast(Dict[str, Any], embedding_metadata)


def get_embedding(
    census_version: str,
    embedding_uri: str,
    obs_soma_joinids: npt.NDArray[np.int64] | pa.Array,
    context: soma.options.SOMATileDBContext | None = None,
) -> npt.NDArray[np.float32]:
    """Read cell (obs) embeddings and return as a dense :class:`numpy.ndarray`. Any cells without
    an embedding will return NaN values.

    Args:
        census_version:
            The Census version tag, e.g., ``"2023-12-15"``. Used to verify that the contents of
            the embedding contain embedded cells from the same Census version.
        embedding_uri:
            The URI containing the embedding data.
        obs_soma_joinids:
            The slice of the embedding to fetch and return.
        context:
            A custom :class:`tiledbsoma.SOMATileDBContext` which will be used to open the SOMA object.
            Optional, defaults to ``None``.

    Returns:
        A :class:`numpy.ndarray` containing the embeddings. Embeddings are positionally
        indexed by the ``obs_soma_joinids``. In other words, the cell identified by
        ``obs_soma_joinids[i]`` corresponds to the ``ith`` position in the returned
        :class:`numpy.ndarray`.

    Raises:
        ValueError: if the Census and embedding are mismatched.

    Lifecycle:
        experimental

    Examples:
        >>> obs_somaids_to_fetch = np.array([10,11], dtype=np.int64)
        >>> emb = cellxgene_census.experimental.get_embedding('2023-12-15', embedding_uri, obs_somaids_to_fetch)
        >>> emb.shape
        (2, 200)
        >>> emb[:, 0:4]
        array([[ 0.02954102,  1.0390625 , -0.14550781, -0.40820312],
            [-0.00224304,  1.265625  ,  0.05883789, -0.7890625 ]],
            dtype=float32)

    """
    if isinstance(obs_soma_joinids, (pa.Array, pa.ChunkedArray, pd.Series)):
        obs_soma_joinids = obs_soma_joinids.to_numpy()
    assert isinstance(obs_soma_joinids, np.ndarray)
    if obs_soma_joinids.dtype != np.int64:
        raise TypeError("obs_soma_joinids must be array of int64")

    # Allow the user to override context for exceptional cases (e.g. the aws region)
    context = context or get_default_soma_context()

    # Attempt to resolve census version aliases
    census_directory = get_census_version_directory()
    resolved_census_version = census_directory.get(census_version, None)

    with soma.open(embedding_uri, context=context) as E:
        embedding_metadata = json.loads(E.metadata["CxG_embedding_info"])

        if resolved_census_version is None:
            warnings.warn(
                "Unable to determine Census version - skipping validation of Census and embedding version.",
                stacklevel=1,
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

        obs_indexer = soma.tiledbsoma_build_index(obs_soma_joinids, context=E.context)
        for tbl in E.read(coords=(obs_soma_joinids,)).tables():
            obs_idx = obs_indexer.get_indexer(tbl.column("soma_dim_0").to_numpy())
            feat_idx = tbl.column("soma_dim_1").to_numpy()
            emb = tbl.column("soma_data")

            indices = obs_idx * E.shape[1] + feat_idx
            np.put(embedding.reshape(-1), indices, emb)

    return embedding
