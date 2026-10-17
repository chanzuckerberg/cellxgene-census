"""Offline, native SOMA integration tests for biologically constrained search."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pytest
import requests
import tiledbsoma as soma
from scipy.spatial.distance import cdist

from cellxgene_census._open import get_default_soma_context
from cellxgene_census.experimental import _embedding_search_exact as exact
from cellxgene_census.experimental import find_nearest_obs

VERSION = "2025-11-08"
ORGANISM = "homo_sapiens"
pytestmark = pytest.mark.experimental


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict[str, Any]]:
    def no_network(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Offline search test attempted an HTTP request")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    context = get_default_soma_context(tiledb_config={"sm.compute_concurrency_level": 2, "sm.io_concurrency_level": 2})
    original_context = get_default_soma_context

    def context_for_test(tiledb_config: dict[str, Any]) -> Any:
        return original_context(
            tiledb_config={**tiledb_config, "sm.compute_concurrency_level": 2, "sm.io_concurrency_level": 2}
        )

    monkeypatch.setattr(exact, "get_default_soma_context", context_for_test)

    def create(
        values: npt.NDArray[Any],
        *,
        ids: npt.NDArray[Any] | None = None,
        tissues: list[str] | None = None,
        primary: list[bool] | None = None,
        omitted_rows: tuple[int, ...] = (),
        partial_rows: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        values = np.asarray(values, dtype=np.float32)
        n, d = values.shape
        ids = np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        metadata = {
            "id": "test-embedding",
            "embedding_name": "scvi",
            "experiment_name": ORGANISM,
            "census_version": VERSION,
            "measurement_name": "RNA",
            "data_type": "obs_embedding",
            "n_features": d,
            "relative_uri": "embedding",
            "indexes": [],
        }
        census: Any
        with soma.Collection.create(str(tmp_path / "census"), context=context) as census:
            data = census.add_new_collection("census_data")
            experiment = data.add_new_collection(ORGANISM, kind=soma.Experiment)
            ms = experiment.add_new_collection("ms")
            ms.add_new_collection("RNA", kind=soma.Measurement)
            table = pa.table(
                {
                    "soma_joinid": ids,
                    "tissue": tissues or ["lung"] * n,
                    "is_primary_data": primary if primary is not None else [True] * n,
                }
            )
            obs = experiment.add_new_dataframe("obs", schema=table.schema, domain=[(0, int(ids.max()))])
            obs.write(table)
        with soma.SparseNDArray.create(
            str(tmp_path / "embedding"), type=pa.float32(), shape=(int(ids.max()) + 1, d), context=context
        ) as embedding:
            embedding.metadata["CxG_embedding_info"] = json.dumps(metadata)
            rows = np.repeat(ids, d)
            cols = np.tile(np.arange(d, dtype=np.int64), n)
            keep = ~np.isin(rows, ids[list(omitted_rows)])
            keep &= ~(np.isin(rows, ids[list(partial_rows)]) & (cols == d - 1))
            embedding.write(
                pa.table({"soma_dim_0": rows[keep], "soma_dim_1": cols[keep], "soma_data": values.ravel()[keep]})
            )
        monkeypatch.setattr(exact, "get_census_version_description", lambda version: {"release_build": VERSION})
        monkeypatch.setattr(
            exact,
            "_get_census_mirrors",
            lambda: {
                "default": "local",
                "local": {"embeddings_base_uri": tmp_path.as_uri() + "/", "region": "us-west-2", "provider": "file"},
            },
        )
        monkeypatch.setattr(
            exact,
            "open_soma",
            lambda **kwargs: soma.Collection.open(str(tmp_path / "census"), context=kwargs["context"]),
        )
        return metadata

    return create


def search(metadata: dict[str, Any], query: npt.NDArray[Any], **kwargs: Any) -> Any:
    query = np.asarray(query)
    adata = ad.AnnData(np.zeros((len(query), 0), dtype=np.float32))
    adata.obsm["scvi"] = query
    return find_nearest_obs(
        "scvi",
        ORGANISM,
        VERSION,
        adata,
        embedding_metadata=metadata,
        obs_value_filter=kwargs.pop("obs_value_filter", "is_primary_data == True"),
        **kwargs,
    )


def oracle(query: npt.NDArray[Any], values: npt.NDArray[Any], ids: npt.NDArray[Any], k: int) -> tuple[Any, Any]:
    distances = cdist(query, values.astype(np.float32), metric="sqeuclidean")
    order = np.array([np.lexsort((ids, row))[:k] for row in distances])
    return np.take_along_axis(distances, order, axis=1).astype(np.float32), ids[order]


@pytest.mark.parametrize("reference_rows,query_rows", [(1, 1), (3, 2), (8, 7), (8192, 256)])
def test_exact_oracle_and_chunk_invariance(
    corpus: Any, monkeypatch: pytest.MonkeyPatch, reference_rows: int, query_rows: int
) -> None:
    rng = np.random.default_rng(91)
    values = rng.normal(size=(37, 5)).astype(np.float32)
    queries = rng.normal(size=(11, 5)).astype(np.float32)
    ids = np.arange(37, dtype=np.int64) * 13 + 9
    primary = [i % 3 != 0 for i in range(37)]
    metadata = corpus(values, ids=ids, primary=primary)
    monkeypatch.setattr(exact, "_MAX_REFERENCE_ROWS", reference_rows)
    monkeypatch.setattr(exact, "_MAX_QUERY_ROWS", query_rows)
    result = search(metadata, queries, k=5)
    expected_d, expected_i = oracle(queries, values[primary], ids[primary], 5)
    np.testing.assert_array_equal(result.neighbor_ids, expected_i)
    np.testing.assert_array_equal(result.distances, expected_d)
    assert result.distances.dtype == np.float32
    assert result.neighbor_ids.dtype == np.uint64


def test_filter_before_ranking_beyond_fixed_oversampling(corpus: Any) -> None:
    # Even 50*k global neighbors are ineligible; post-filtering them returns none.
    values = np.arange(305, dtype=np.float32).reshape(-1, 1)
    metadata = corpus(values, primary=[False] * 300 + [True] * 5)
    result = search(metadata, np.array([[0.0]], dtype=np.float32), k=3)
    np.testing.assert_array_equal(result.neighbor_ids, [[300, 301, 302]])
    np.testing.assert_array_equal(result.distances, [[90000, 90601, 91204]])


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("tissue == 'lung' and is_primary_data == True", [20]),
        ("tissue in ['lung', 'brain'] and is_primary_data == True", [20]),
        ("tissue == 'heart' or is_primary_data == False", [4]),
        ("tissue != 'lung'", [4]),
        ("soma_joinid < 32 and is_primary_data == True", [20]),
    ],
)
def test_native_filter_language(corpus: Any, predicate: str, expected: list[int]) -> None:
    metadata = corpus(
        np.array([[1.0], [2.0], [3.0]]),
        ids=np.array([4, 20, 99]),
        tissues=["heart", "lung", "brain"],
        primary=[False, True, True],
    )
    result = search(metadata, np.array([[0.0]]), k=1, obs_value_filter=predicate)
    np.testing.assert_array_equal(result.neighbor_ids[0], expected)


def test_ties_across_batches(corpus: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = corpus(np.ones((17, 2)), ids=np.arange(17, dtype=np.int64) * 5 + 11)
    monkeypatch.setattr(exact, "_MAX_REFERENCE_ROWS", 2)
    result = search(metadata, np.array([[0.0, 0.0]]), k=4)
    np.testing.assert_array_equal(result.neighbor_ids, [[11, 16, 21, 26]])
    np.testing.assert_array_equal(result.distances, [[2, 2, 2, 2]])


def test_stable_distance_at_large_common_offset(corpus: Any) -> None:
    # Float32 stored values are distinct here; norm-sum-dot is vulnerable to cancellation.
    base = np.float32(2**24)
    values = np.array([[base + 8, base], [base + 2, base], [base + 4, base]], dtype=np.float32)
    metadata = corpus(values)
    result = search(metadata, np.array([[base, base]], dtype=np.float32), k=3)
    np.testing.assert_array_equal(result.neighbor_ids, [[1, 2, 0]])
    np.testing.assert_array_equal(result.distances, [[4, 16, 64]])


def test_extended_precision_query_chunk_invariance(corpus: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = corpus(np.array([[0, 1], [1, 0]]), ids=np.array([1, 2]))
    query = np.array([[-(2**-60), -(2**-59)]], dtype=np.longdouble)
    results = []
    for block_size in (1, 8192):
        monkeypatch.setattr(exact, "_MAX_REFERENCE_ROWS", block_size)
        results.append(search(metadata, query, k=1))
    np.testing.assert_array_equal(results[0].neighbor_ids, results[1].neighbor_ids)
    np.testing.assert_array_equal(results[0].distances, results[1].distances)


@pytest.mark.skipif(bool(np.finfo(np.longdouble).max <= np.finfo(np.float64).max), reason="No extended float range")
def test_query_outside_float64_range(corpus: Any) -> None:
    metadata = corpus(np.array([[0.0]]))
    query = np.array([[np.finfo(np.longdouble).max]], dtype=np.longdouble)
    with pytest.raises(ValueError, match="only finite"):
        search(metadata, query, k=1)


def test_missing_partial_and_nonfinite_embeddings(corpus: Any) -> None:
    metadata = corpus(
        np.array([[0, 0], [1, 1], [np.nan, 2], [np.inf, 0], [3, 4]]), omitted_rows=(0,), partial_rows=(1,)
    )
    result = search(metadata, np.array([[0.0, 0.0]]), k=1)
    np.testing.assert_array_equal(result.neighbor_ids, [[4]])
    np.testing.assert_array_equal(result.distances, [[25]])


@pytest.mark.parametrize("k,predicate,count", [(3, "is_primary_data == True", 2), (1, "tissue == 'absent'", 0)])
def test_insufficient_finite_candidates(corpus: Any, k: int, predicate: str, count: int) -> None:
    metadata = corpus(np.array([[0.0], [np.nan], [1.0]]))
    with pytest.raises(ValueError, match=f"Only {count} finite embeddings"):
        search(metadata, np.array([[0.0]]), k=k, obs_value_filter=predicate)


@pytest.mark.parametrize("query", [np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-np.inf]])])
def test_nonfinite_query(corpus: Any, query: Any) -> None:
    metadata = corpus(np.array([[0.0]]))
    with pytest.raises(ValueError, match="only finite"):
        search(metadata, query, k=1)


@pytest.mark.parametrize("predicate", ["", "   ", 5])
def test_invalid_filter_argument(corpus: Any, predicate: Any) -> None:
    metadata = corpus(np.array([[0.0]]))
    with pytest.raises(ValueError, match="nonempty"):
        search(metadata, np.array([[0.0]]), k=1, obs_value_filter=predicate)


def test_invalid_native_predicate(corpus: Any) -> None:
    metadata = corpus(np.array([[0.0]]))
    with pytest.raises((soma.SOMAError, ValueError)):
        search(metadata, np.array([[0.0]]), k=1, obs_value_filter="unknown_field == 'x'")


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("census_version", "other", "census_version"),
        ("experiment_name", "mus_musculus", "experiment_name"),
        ("data_type", "var_embedding", "obs_embedding"),
        ("embedding_name", "other", "embedding_name"),
    ],
)
def test_metadata_validation(corpus: Any, key: str, value: str, message: str) -> None:
    metadata = corpus(np.array([[0.0]]))
    metadata[key] = value
    with pytest.raises((ValueError, AssertionError), match=message if key != "embedding_name" else None):
        search(metadata, np.array([[0.0]]), k=1)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(corpus: Any, k: Any) -> None:
    metadata = corpus(np.array([[0.0]]))
    with pytest.raises(ValueError, match="positive integer"):
        search(metadata, np.array([[0.0]]), k=k)


def test_distance_overflow(corpus: Any) -> None:
    metadata = corpus(np.array([[1e30]], dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        search(metadata, np.array([[0.0]]), k=1)


@pytest.mark.parametrize("k", [1, 10])
def test_empty_query(corpus: Any, monkeypatch: pytest.MonkeyPatch, k: int) -> None:
    metadata = corpus(np.array([[0.0], [1.0]]))

    def no_candidate_io(*args: Any, **kwargs: Any) -> None:
        pytest.fail("An empty query must not open or scan the candidate corpus")

    monkeypatch.setattr(exact, "open_soma", no_candidate_io)
    monkeypatch.setattr(exact, "_get_census_mirrors", no_candidate_io)
    result = search(metadata, np.empty((0, 1)), k=k)
    assert result.neighbor_ids.shape == result.distances.shape == (0, k)
    assert result.distances.dtype == np.float32
    assert result.neighbor_ids.dtype == np.uint64


def test_workspace_rejects_oversized_results() -> None:
    with pytest.raises(ValueError, match="too small"):
        exact._search_block_sizes(10000, 50, 1000, 1000000)


def test_unfiltered_ann_route_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiledb.vector_search as vs

    from cellxgene_census.experimental import _embedding_search as search_module

    captured = {}
    expected_distances = np.array([[2.0]], dtype=np.float32)
    expected_ids = np.array([[19]], dtype=np.uint64)

    class FakeIndex:
        def __init__(self, **kwargs: Any) -> None:
            captured["open"] = kwargs

        def query(self, query: Any, **kwargs: Any) -> Any:
            captured["query"] = kwargs
            return expected_distances, expected_ids

    monkeypatch.setattr(vs.ivf_flat_index, "IVFFlatIndex", FakeIndex)
    monkeypatch.setattr(
        search_module, "_resolve_embedding_index", lambda *args, **kwargs: ("file:///index", "us-west-2")
    )
    metadata = {"embedding_name": "scvi", "n_features": 1}
    result = search(metadata, np.array([[0.0]]), obs_value_filter=None, k=1, nprobe=3, memory_GiB=2, nthreads=1)
    assert result.distances is expected_distances
    assert result.neighbor_ids is expected_ids
    assert captured["query"] == {"k": 1, "nprobe": 3, "nthreads": 1}
    assert captured["open"]["memory_budget"] == 2 * 2**30 // 4


def test_exact_rejects_index_kwargs(corpus: Any) -> None:
    metadata = corpus(np.array([[0.0]]))
    with pytest.raises(ValueError, match="Index query kwargs"):
        search(metadata, np.array([[0.0]]), k=1, nthreads=2)


def test_dimension_mismatch(corpus: Any) -> None:
    metadata = corpus(np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="features, expected"):
        search(metadata, np.array([[0.0]]), k=1)


@pytest.mark.parametrize("memory", [0, -1, True, 0.1])
def test_invalid_memory(corpus: Any, memory: Any) -> None:
    metadata = corpus(np.array([[0.0]]))
    with pytest.raises(ValueError, match="memory_GiB"):
        search(metadata, np.array([[0.0]]), k=1, memory_GiB=memory)


def test_streamed_embedding_requests_are_bounded(corpus: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = corpus(np.arange(150, dtype=np.float32).reshape(50, 3))
    monkeypatch.setattr(exact, "_MAX_REFERENCE_ROWS", 7)
    monkeypatch.setattr(exact, "_MAX_QUERY_ROWS", 2)
    read = exact._read_embedding_rows
    distance = cdist
    requested = []
    blocks = []

    def read_spy(embedding: Any, ids: Any, n_features: Any) -> Any:
        requested.append(len(ids))
        return read(embedding, ids, n_features)

    def distance_spy(queries: Any, values: Any, **kwargs: Any) -> Any:
        blocks.append((len(queries), len(values)))
        return distance(queries, values, **kwargs)

    monkeypatch.setattr(exact, "_read_embedding_rows", read_spy)
    monkeypatch.setattr(exact, "cdist", distance_spy)
    search(metadata, np.zeros((5, 3)), k=3)
    assert len(requested) == 8
    assert max(requested) <= 7
    assert all(q <= 2 and r <= 7 for q, r in blocks)


@pytest.mark.parametrize("key,value", [("id", "other-submission"), ("data_type", "var_embedding"), ("n_features", 7)])
def test_stored_metadata_identity_mismatch(corpus: Any, tmp_path: Path, key: str, value: Any) -> None:
    metadata = corpus(np.array([[0.0]]))
    with soma.SparseNDArray.open(str(tmp_path / "embedding"), mode="w") as embedding:
        stored = json.loads(embedding.metadata["CxG_embedding_info"])
        stored[key] = value
        embedding.metadata["CxG_embedding_info"] = json.dumps(stored)
    with pytest.raises(ValueError, match=key):
        search(metadata, np.array([[0.0]]), k=1)


def test_exact_distance_units_match_native_ivfflat(corpus: Any, tmp_path: Path) -> None:
    import subprocess
    import sys

    # Isolate TileDB's process-global default context before importing vector_search.
    # Ingest/query are explicitly LOCAL, with four synthetic vectors and no service.
    script = """
import sys
import numpy as np
import tiledb
config = {"sm.compute_concurrency_level": "2", "sm.io_concurrency_level": "2"}
tiledb.default_ctx(config)
import tiledb.vector_search as vs
vectors = np.array([[3, 4], [0, 6], [8, 0], [5, 12]], dtype=np.float32)
ids = np.array([91, 37, 55, 12], dtype=np.uint64)
index = vs.ingest(index_type="IVF_FLAT", index_uri=sys.argv[1], input_vectors=vectors,
                  external_ids=ids, partitions=1, workers=1, config=config,
                  mode=vs.ingestion.Mode.LOCAL, training_sample_size=4)
distances, ids = index.query(np.array([[0, 0]], dtype=np.float32), k=4, nprobe=1)
np.testing.assert_array_equal(distances, [[25, 36, 64, 169]])
np.testing.assert_array_equal(ids, [[91, 37, 55, 12]])
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "ivf")], check=True, capture_output=True, text=True)
    values = np.array([[3, 4], [0, 6], [8, 0], [5, 12]], dtype=np.float32)
    metadata = corpus(values, ids=np.array([91, 37, 55, 12], dtype=np.int64))
    result = search(metadata, np.array([[0, 0]], dtype=np.float32), k=4)
    np.testing.assert_array_equal(result.distances, [[25, 36, 64, 169]])
    np.testing.assert_array_equal(result.neighbor_ids, [[91, 37, 55, 12]])
