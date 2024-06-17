"""Nearest-neighbor search based on vector index of Census embeddings."""

from contextlib import ExitStack
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledb.vector_search as vs
import tiledbsoma as soma

from .._open import DEFAULT_TILEDB_CONFIGURATION, open_soma

CENSUS_EMBEDDINGS_INDEX_URI_FSTR = (
    "s3://cellxgene-contrib-public/contrib/cell-census/soma/{census_version}/indexes/{embedding_id}"
)
CENSUS_EMBEDDINGS_INDEX_REGION = "us-west-2"


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
    embedding_metadata: Dict[str, Any],
    query: ad.AnnData,
    k: int = 10,
    nprobe: int = 100,
    memory_GiB: int = 4,
    **kwargs: Dict[str, Any],
) -> NeighborObs:
    """Search Census for similar obs (cells) based on nearest neighbors in embedding space.

    Args:
        embedding_metadata:
            Information about the embedding to search, as found by
            :func:`get_embedding_metadata_by_name`.
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
    """
    embedding_name = embedding_metadata["embedding_name"]
    n_features = embedding_metadata["n_features"]

    # validate query (expected obsm layer exists with the expected dimensionality)
    if embedding_name not in query.obsm:
        raise ValueError(f"Query does not have the expected layer {embedding_name}")
    if query.obsm[embedding_name].shape[1] != n_features:
        raise ValueError(
            f"Query embedding {embedding_name} has {query.obsm[embedding_name].shape[1]} features, expected {n_features}"
        )

    # formulate index URI and run query
    index_uri = CENSUS_EMBEDDINGS_INDEX_URI_FSTR.format(
        census_version=embedding_metadata["census_version"], embedding_id=embedding_metadata["id"]
    )
    config = {k: str(v) for k, v in DEFAULT_TILEDB_CONFIGURATION.items()}
    config["vfs.s3.region"] = CENSUS_EMBEDDINGS_INDEX_REGION
    memory_vectors = memory_GiB * (2**30) // (4 * n_features)  # number of float32 vectors
    index = vs.ivf_flat_index.IVFFlatIndex(uri=index_uri, config=config, memory_budget=memory_vectors)
    distances, neighbor_ids = index.query(query.obsm[embedding_name], k=k, nprobe=nprobe, **kwargs)

    return NeighborObs(distances=distances, neighbor_ids=neighbor_ids)


def predict_obs_metadata(
    embedding_metadata: Dict[str, Any],
    neighbors: NeighborObs,
    column_names: Sequence[str],
    experiment: Optional[soma.Experiment] = None,
) -> pd.DataFrame:
    """Predict obs metadata attributes for the query cells based on the embedding nearest neighbors.

    Args:
        embedding_metadata:
            Information about the embedding searched, as found by
            :func:`get_embedding_metadata_by_name`.
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
            census = cleanup.enter_context(open_soma(census_version=embedding_metadata["census_version"]))
            experiment = census["census_data"][embedding_metadata["experiment_name"]]

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
        out: Dict[str, List[Any]] = {}
        for i in range(neighbors.neighbor_ids.shape[0]):
            neighbors_i = neighbor_obs.loc[neighbors.neighbor_ids[i]]
            for col in column_names:
                col_value_counts = neighbors_i[col].value_counts(normalize=True)
                out.setdefault(col, []).append(col_value_counts.idxmax())
                out.setdefault(col + "_confidence", []).append(col_value_counts.max())

    return pd.DataFrame.from_dict(out)
