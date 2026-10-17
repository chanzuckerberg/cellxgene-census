"""Filter-first exact neighbor search over bounded batches of stored embeddings."""

import json
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import tiledbsoma as soma
from scipy.spatial.distance import cdist

from .._experiment import _get_experiment_name
from .._open import get_default_soma_context, open_soma
from .._release_directory import CensusMirror, _get_census_mirrors, get_census_version_description
from .._util import _uri_join

_MAX_REFERENCE_ROWS = 8192
_MAX_QUERY_ROWS = 256
_IO_BUFFER_BYTES = 8 * 2**20
_IO_RESERVE_BYTES = 8 * _IO_BUFFER_BYTES


def _search_block_sizes(n_queries: int, n_features: int, k: int, budget: int) -> tuple[int, int]:
    """Budget numerical arrays and leave I/O headroom, not a hard native/Arrow cap."""
    # float64 running distances, uint64 IDs, and the final float32 distance copy.
    available = budget - n_queries * k * 20 - _IO_RESERVE_BYTES
    if available < 128 * (n_features + k + 1):
        raise ValueError("memory_GiB is too small for the query results and search workspace")
    # Sparse embedding reads include dimension columns and temporary index arrays.
    reference_rows = min(_MAX_REFERENCE_ROWS, max(1, available // (128 * (n_features + 1))))
    query_rows = min(_MAX_QUERY_ROWS, max(1, available // (128 * (reference_rows + k + n_features))))
    return reference_rows, query_rows


def _read_embedding_rows(
    embedding: soma.SparseNDArray, ids: npt.NDArray[np.int64], n_features: int
) -> npt.NDArray[np.float32]:
    """Match get_embedding's NaN semantics without materializing an entire selection."""
    values = np.full((len(ids), n_features), np.nan, dtype=np.float32)
    indexer = soma.IntIndexer(ids, context=embedding.context)
    for table in embedding.read(coords=(ids,)).tables():
        rows = indexer.get_indexer(table.column("soma_dim_0").to_numpy())
        cols = table.column("soma_dim_1").to_numpy()
        values[rows, cols] = table.column("soma_data").to_numpy()
    return values


def _find_nearest_obs_exact(
    queries: object,
    *,
    organism: str,
    census_version: str,
    embedding_metadata: dict[str, Any],
    obs_value_filter: str,
    k: int,
    memory_GiB: int,
    mirror: str | None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint64]]:
    """Return exact squared-L2 neighbors after native SOMA predicate evaluation."""
    if not isinstance(obs_value_filter, str) or not obs_value_filter.strip():
        raise ValueError("obs_value_filter must be a nonempty SOMA value filter")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(memory_GiB, bool) or not isinstance(memory_GiB, int) or memory_GiB < 1:
        raise ValueError("memory_GiB must be a positive integer")
    if not isinstance(queries, np.ndarray) or queries.ndim != 2 or queries.dtype.kind not in "fiu":
        raise ValueError("Query embeddings must be a two-dimensional real numeric NumPy array")
    n_queries, n_features = queries.shape
    if n_features == 0:
        raise ValueError("Query embeddings must contain at least one feature")
    version = get_census_version_description(census_version)["release_build"]
    experiment_name = _get_experiment_name(organism)
    if embedding_metadata["census_version"] != version:
        raise ValueError("Census and embedding mismatch - census_version not equal")
    if embedding_metadata["experiment_name"] != experiment_name:
        raise ValueError("Census and embedding mismatch - experiment_name not equal")
    if embedding_metadata.get("data_type") != "obs_embedding":
        raise ValueError("Exact cell search requires an obs_embedding")
    if n_queries == 0:
        return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.uint64)

    budget = memory_GiB * 2**30
    reference_rows, query_rows = _search_block_sizes(n_queries, n_features, k, budget)
    for start in range(0, n_queries, query_rows):
        with np.errstate(over="ignore"):
            query_block = np.asarray(queries[start : start + query_rows], dtype=np.float64)
        if not np.isfinite(query_block).all():
            raise ValueError("Query embeddings must contain only finite values")
    del query_block

    mirrors = _get_census_mirrors()
    selected_mirror = mirror or cast(str, mirrors["default"])
    mirror_info = cast(CensusMirror, mirrors[selected_mirror])
    uri = _uri_join(mirror_info["embeddings_base_uri"], embedding_metadata["relative_uri"])
    # Do not inherit Census's 1-GiB initial read buffers for a streaming search.
    buffer_bytes = min(_IO_BUFFER_BYTES, budget // 16)
    context = get_default_soma_context(
        tiledb_config={
            "vfs.s3.region": mirror_info["region"],
            "soma.init_buffer_bytes": buffer_bytes,
            "py.init_buffer_bytes": buffer_bytes,
        }
    )

    distances = np.full((n_queries, k), np.inf, dtype=np.float64)
    neighbor_ids = np.full((n_queries, k), np.iinfo(np.uint64).max, dtype=np.uint64)
    n_valid = 0
    with (
        open_soma(census_version=version, mirror=selected_mirror, context=context) as census,
        soma.open(uri, mode="r", soma_type=soma.SparseNDArray, context=context) as embedding,
    ):
        stored_metadata = json.loads(embedding.metadata["CxG_embedding_info"])
        for key in (
            "id",
            "data_type",
            "n_features",
            "census_version",
            "experiment_name",
            "measurement_name",
            "embedding_name",
        ):
            if stored_metadata[key] != embedding_metadata[key]:
                raise ValueError(f"Census and embedding mismatch - {key} not equal")
        if embedding.shape[1] != n_features:
            raise ValueError("Stored embedding dimensionality does not match query embeddings")
        experiment = census["census_data"][experiment_name]
        if embedding_metadata["measurement_name"] not in experiment.ms:
            raise ValueError("Census and embedding mismatch - measurement_name does not exist")
        # Stream the metadata selection too: never collect all eligible soma_joinids.
        for table in experiment.obs.read(column_names=["soma_joinid"], value_filter=obs_value_filter):
            # Some SOMA versions also return predicate columns. Retain only IDs;
            # native Arrow batch sizes remain outside the numerical memory budget.
            table = table.select(["soma_joinid"])
            for offset in range(0, len(table), reference_rows):
                ids = table.column("soma_joinid").slice(offset, reference_rows).to_numpy()
                values = _read_embedding_rows(embedding, ids, n_features)
                finite = np.isfinite(values).all(axis=1)
                ids, values = ids[finite], values[finite]
                n_valid += len(ids)
                if not len(ids):
                    continue
                for start in range(0, n_queries, query_rows):
                    stop = min(start + query_rows, n_queries)
                    # Direct coordinate differences avoid norm/dot cancellation at
                    # large common offsets. Accumulate in float64 before output cast.
                    query_block = np.asarray(queries[start:stop], dtype=np.float64)
                    block = cdist(query_block, values, metric="sqeuclidean")
                    for row, candidate_distances in enumerate(block, start):
                        all_distances = np.concatenate((distances[row], candidate_distances))
                        all_ids = np.concatenate((neighbor_ids[row], ids.astype(np.uint64)))
                        # Full lexicographic selection also handles >k equal-distance
                        # candidates spanning chunks; argpartition alone does not.
                        order = np.lexsort((all_ids, all_distances))[:k]
                        distances[row], neighbor_ids[row] = all_distances[order], all_ids[order]

    if n_valid < k:
        raise ValueError(f"Only {n_valid} finite embeddings match obs_value_filter; at least k={k} are required")
    if not np.isfinite(distances).all() or np.any(distances > np.finfo(np.float32).max):
        raise ValueError("Squared distances cannot be represented as finite float32 values")
    return distances.astype(np.float32), neighbor_ids
