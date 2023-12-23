import numpy as np
import pytest
import scipy.sparse as sparse
import tiledbsoma as soma

import cellxgene_census
from cellxgene_census import get_default_soma_context
from cellxgene_census.experimental.util import X_sparse_iter


@pytest.fixture
def small_mem_context() -> soma.SOMATileDBContext:
    """used to keep memory usage smaller for GHA runners."""
    return get_default_soma_context(tiledb_config={"soma.init_buffer_bytes": 32 * 1024**2})


@pytest.mark.experimental
@pytest.mark.live_corpus
def test_X_sparse_iter(small_mem_context: soma.SOMATileDBContext) -> None:
    with cellxgene_census.open_soma(census_version="latest", context=small_mem_context) as census:
        obs_filter = """is_primary_data == True and tissue_general == 'tongue'"""
        expt = census["census_data"]["mus_musculus"]
        Xraw = expt.ms["RNA"].X["raw"]
        with expt.axis_query(measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_filter)) as query:
            for (obs_ids, var_ids), X_chunk in X_sparse_iter(query, fmt="csr"):
                assert X_chunk.shape[0] == len(obs_ids)
                assert X_chunk.shape[1] == len(var_ids)
                assert obs_ids.dtype == np.int64
                assert var_ids.dtype == np.int64
                assert sparse.isspmatrix_csr(X_chunk)

            # test reindex_sparse_axis=False
            for (obs_ids, _), X_chunk in X_sparse_iter(query, fmt="csr", reindex_sparse_axis=False):
                assert X_chunk.shape[0] == len(obs_ids)
                assert X_chunk.shape[1] == Xraw.shape[1]
                assert sparse.isspmatrix_csr(X_chunk)


@pytest.mark.experimental
@pytest.mark.live_corpus
def test_X_sparse_iter_unsupported(small_mem_context: soma.SOMATileDBContext) -> None:
    with cellxgene_census.open_soma(census_version="latest", context=small_mem_context) as census:
        with census["census_data"]["mus_musculus"].axis_query(measurement_name="RNA") as query:
            with pytest.raises(ValueError):
                next(X_sparse_iter(query, axis=1))

            with pytest.raises(ValueError):
                next(X_sparse_iter(query, fmt="foobar"))  # type: ignore[arg-type]
