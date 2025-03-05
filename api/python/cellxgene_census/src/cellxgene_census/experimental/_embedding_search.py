"""Nearest-neighbor search based on vector index of Census embeddings."""

from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any, NamedTuple, cast

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledbsoma as soma
from scipy import sparse

from .._experiment import _get_experiment_name
from .._open import DEFAULT_TILEDB_CONFIGURATION, open_soma
from .._release_directory import CensusMirror, _get_census_mirrors
from .._util import _uri_join
from ._embedding import get_embedding_metadata_by_name


class NeighborObs(NamedTuple):
    """Results of nearest-neighbor search for Census obs embeddings."""

    distances: npt.NDArray[np.float32]
    """
    Distances to the nearest neighbors for each query obs embedding (q by k, where q is the number
    of query embeddings and k is the desired number of neighbors). The distance metric is
    implementation-dependent.
    """

    neighbor_ids: npt.NDArray[np.int64]
    """
    obs soma_joinid's of the nearest neighbors for each query embedding (q by k).
    """


def find_nearest_obs(
    embedding_name: str,
    organism: str,
    census_version: str,
    query: ad.AnnData,
    *,
    k: int = 10,
    nprobe: int = 100,
    memory_GiB: int = 4,
    mirror: str | None = None,
    embedding_metadata: dict[str, Any] | None = None,
    **kwargs: dict[str, Any],
) -> NeighborObs:
    """Search Census for similar obs (cells) based on nearest neighbors in embedding space.

    Args:
        embedding_name, organism, census_version:
            Identify the embedding to search, as in :func:`get_embedding_metadata_by_name`.
        query:
            AnnData object with an obsm layer embedding the query cells. The obsm layer name
            matches ``embedding_metadata["embedding_name"]`` (e.g. scvi, geneformer). The layer
            shape matches the number of query cells and the number of features in the embedding.
        k:
            Number of nearest neighbors to return for each query obs.
        nprobe:
            Sensitivity parameter; defaults to 100 (roughly N^0.25 where N is the number of Census
            cells) for a thorough search. Decrease for faster but less accurate search.
        memory_GiB:
            Memory budget for the search index, in gibibytes; defaults to 4 GiB.
        mirror:
            Name of the Census mirror to use for the search.
        embedding_metadata:
            The result of `get_embedding_metadata_by_name(embedding_name, organism, census_version)`.
            Supplying this saves a network request for repeated searches.
    """
    import tiledb.vector_search as vs

    if embedding_metadata is None:
        embedding_metadata = get_embedding_metadata_by_name(embedding_name, organism, census_version)
    assert embedding_metadata["embedding_name"] == embedding_name
    n_features = embedding_metadata["n_features"]

    # validate query (expected obsm layer exists with the expected dimensionality)
    if embedding_name not in query.obsm:
        raise ValueError(f"Query does not have the expected layer {embedding_name}")
    if query.obsm[embedding_name].shape[1] != n_features:
        raise ValueError(
            f"Query embedding {embedding_name} has {query.obsm[embedding_name].shape[1]} features, expected {n_features}"
        )

    # formulate index URI and run query
    resolved_index = _resolve_embedding_index(embedding_metadata, mirror=mirror)
    if not resolved_index:
        raise ValueError("No suitable embedding index found for " + embedding_name)
    index_uri, index_region = resolved_index
    config = {k: str(v) for k, v in DEFAULT_TILEDB_CONFIGURATION.items()}
    config["vfs.s3.region"] = index_region
    memory_vectors = memory_GiB * (2**30) // (4 * n_features)  # number of float32 vectors
    index = vs.ivf_flat_index.IVFFlatIndex(uri=index_uri, config=config, memory_budget=memory_vectors)
    distances, neighbor_ids = index.query(query.obsm[embedding_name], k=k, nprobe=nprobe, **kwargs)

    return NeighborObs(distances=distances, neighbor_ids=neighbor_ids)


def _resolve_embedding_index(
    embedding_metadata: dict[str, Any],
    mirror: str | None = None,
) -> tuple[str, str] | None:
    index_metadata = embedding_metadata.get("indexes", None)
    if not index_metadata:
        return None
    # TODO (future): support multiple index [types]
    assert index_metadata[0]["type"] == "IVFFlat", "Only IVFFlat index is supported (update cellxgene_census)"
    mirrors = _get_census_mirrors()
    mirror = mirror or cast(str, mirrors["default"])
    mirror_info = cast(CensusMirror, mirrors[mirror])
    uri = _uri_join(mirror_info["embeddings_base_uri"], index_metadata[0]["relative_uri"])
    return uri, cast(str, mirror_info["region"])


def predict_obs_metadata(
    organism: str,
    census_version: str,
    neighbors: NeighborObs,
    column_names: Sequence[str],
    experiment: soma.Experiment | None = None,
) -> pd.DataFrame:
    """Predict obs metadata attributes for the query cells based on the embedding nearest neighbors.

    Args:
        organism, census_version:
            Embedding information as supplied to :func:`find_nearest_obs`.
        neighbors:
            Results of a :func:`find_nearest_obs` search.
        column_names:
            Desired obs metadata column names. The current implementation is suitable for
            categorical attributes (e.g. cell_type, tissue_general).
        experiment:
            Open handle for the relevant SOMAExperiment, if available (otherwise, will be opened
            internally). e.g. ``census["census_data"]["homo_sapiens"]`` with the relevant Census
            version.

    Returns:
        Pandas DataFrame with the desired column predictions. Additionally, for each predicted
        column ``col``, an additional column ``col_confidence`` with a confidence score between 0
        and 1.
    """
    with ExitStack() as cleanup:
        if experiment is None:
            # open Census transiently
            census = cleanup.enter_context(open_soma(census_version=census_version))
            experiment = census["census_data"][_get_experiment_name(organism)]

        # fetch the desired obs metadata for all of the found neighbors
        neighbor_obs = (
            experiment.obs.read(
                coords=(neighbors.neighbor_ids.flatten(),), column_names=(["soma_joinid"] + list(column_names))
            )
            .concat()
            .to_pandas()
        ).set_index("soma_joinid")

        # step through query cells to generate prediction for each column as the plurality value
        # found among its neighbors, with a confidence score based on the simple fraction (for now)
        # TODO: something more intelligent for numeric columns! also use distances, etc.
        max_joinid = neighbor_obs.index.max()
        out: dict[str, pd.Series[Any]] = {}
        n_queries, n_neighbors = neighbors.neighbor_ids.shape
        indices = np.broadcast_to(np.arange(n_queries), (n_neighbors, n_queries)).T
        g = sparse.csr_matrix(
            (
                np.broadcast_to(1, n_queries * n_neighbors),
                (
                    indices.flatten(),
                    neighbors.neighbor_ids.astype(np.int64).flatten(),
                ),
            ),
            shape=(n_queries, max_joinid + 1),
        )
        for col in column_names:
            col_categorical = neighbor_obs[col].astype("category")
            joinid2category = sparse.coo_matrix(
                (np.broadcast_to(1, len(neighbor_obs)), (neighbor_obs.index, col_categorical.cat.codes)),
                shape=(max_joinid + 1, len(col_categorical.cat.categories)),
            )
            counts = g @ joinid2category
            rel_counts = counts / counts.sum(axis=1)
            out[col] = col_categorical.cat.categories[rel_counts.argmax(axis=1).A.flatten()].astype(object)
            out[f"{col}_confidence"] = rel_counts.max(axis=1).toarray().flatten()

    return pd.DataFrame.from_dict(out)
