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
import requests
import tiledbsoma as soma

from .._open import get_default_soma_context, open_soma
from .._release_directory import CensusVersionDescription, CensusVersionName, get_census_version_directory

CELL_CENSUS_EMBEDDINGS_MANIFEST_URL = "https://contrib.cellxgene.cziscience.com/contrib/cell-census/contributions.json"


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


def _get_embedding(
    census: soma.Collection,
    census_directory: dict[CensusVersionName, CensusVersionDescription],
    census_version: str,
    embedding_uri: str,
    obs_soma_joinids: npt.NDArray[np.int64] | pa.Array,
    context: soma.options.SOMATileDBContext | None = None,
) -> npt.NDArray[np.float32]:
    """Private. Like get_embedding, but accepts a Census object and a Census directory."""
    if isinstance(obs_soma_joinids, (pa.Array, pa.ChunkedArray, pd.Series)):
        obs_soma_joinids = obs_soma_joinids.to_numpy()
    assert isinstance(obs_soma_joinids, np.ndarray)
    if obs_soma_joinids.dtype != np.int64:
        raise TypeError("obs_soma_joinids must be array of int64")

    # Allow the user to override context for exceptional cases (e.g. the aws region)
    context = context or get_default_soma_context()

    # Attempt to resolve census version aliases
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

        obs_indexer = soma.IntIndexer(obs_soma_joinids, context=E.context)
        for tbl in E.read(coords=(obs_soma_joinids,)).tables():
            obs_idx = obs_indexer.get_indexer(tbl.column("soma_dim_0").to_numpy())
            feat_idx = tbl.column("soma_dim_1").to_numpy()
            emb = tbl.column("soma_data")

            indices = obs_idx * E.shape[1] + feat_idx
            np.put(embedding.reshape(-1), indices, emb)

    return embedding


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
    census_directory = get_census_version_directory()

    with open_soma(census_version=census_version, context=context) as census:
        return _get_embedding(
            census, census_directory, census_version, embedding_uri, obs_soma_joinids, context=context
        )


def get_embedding_metadata_by_name(
    embedding_name: str, organism: str, census_version: str, embedding_type: str | None = "obs_embedding"
) -> dict[str, Any]:
    """Return metadata for a specific embedding. If more embeddings match the query parameters,
    the most recent one will be returned.

    Args:
        embedding_name:
            The name of the embedding, e.g. "scvi".
        organism:
            The organism for which the embedding is associated.
        census_version:
            The Census version tag, e.g., ``"2023-12-15"``.
        embedding_type:
            Either "obs_embedding" or "var_embedding". Defaults to "obs_embedding".

    Returns:
        A dictionary containing metadata describing the embedding.

    Raises:
        ValueError: if no embeddings are found for the specified query parameters.

    """
    response = requests.get(CELL_CENSUS_EMBEDDINGS_MANIFEST_URL)
    response.raise_for_status()

    manifest = cast(Dict[str, Dict[str, Any]], response.json())
    embeddings = []
    for _, obj in manifest.items():
        if (
            obj["embedding_name"] == embedding_name
            and obj["experiment_name"] == organism
            and obj["data_type"] == embedding_type
            and obj["census_version"] == census_version
        ):
            embeddings.append(obj)

    if len(embeddings) == 0:
        raise ValueError(f"No embeddings found for {embedding_name}, {organism}, {census_version}, {embedding_type}")

    return sorted(embeddings, key=lambda x: x["submission_date"])[-1]


def get_all_available_embeddings(census_version: str) -> list[dict[str, Any]]:
    """Return a dictionary of all available embeddings for a given Census version.

    Args:
        census_version:
            The Census version tag, e.g., ``"2023-12-15"``.

    Returns:
        A list of dictionaries, each containing metadata describing an available embedding.

    Examples:
        >>> get_all_available_embeddings('2023-12-15')
        [{
            'experiment_name': 'experiment_1',
            'measurement_name': 'RNA',
            'organism': "homo_sapiens",
            'census_version': '2023-12-15',
            'n_embeddings': 1000,
            'n_features': 200,
            'uri': 's3://bucket/embedding_1'
        }]

    """
    response = requests.get(CELL_CENSUS_EMBEDDINGS_MANIFEST_URL)
    response.raise_for_status()

    embeddings = []
    manifest = response.json()
    for _, obj in manifest.items():
        if obj["census_version"] == census_version:
            embeddings.append(obj)

    return embeddings


def get_all_census_versions_with_embedding(
    embedding_name: str, organism: str, embedding_type: str | None = "obs_embedding"
) -> list[str]:
    """Get a list of all census versions that contain a specific embedding.

    Args:
        embedding_name:
            The name of the embedding, e.g. "scvi".
        organism:
            The organism for which the embedding is associated.
        embedding_type:
            The type of embedding. Defaults to "obs_embedding".

    Returns:
        A list of census versions that contain the specified embedding.
    """
    response = requests.get(CELL_CENSUS_EMBEDDINGS_MANIFEST_URL)
    response.raise_for_status()

    manifest = response.json()
    return sorted(
        {
            obj["census_version"]
            for obj in manifest.values()
            if obj["embedding_name"] == embedding_name
            and obj["experiment_name"] == organism
            and obj["data_type"] == embedding_type
        }
    )
