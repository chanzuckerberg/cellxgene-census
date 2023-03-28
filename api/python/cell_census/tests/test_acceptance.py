"""
Acceptance tests for the Census.

NOTE: those marked `expensive` are not run in the CI as they are, well, expensive.
Several of them will not run to completion except on VERY large hosts.

"""
import collections.abc
from typing import Optional

import cell_census
import pyarrow as pa
import pytest
import tiledb
import tiledbsoma as soma


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
def test_load_axes(organism: str) -> None:
    # Verify axes can be loaded into a Pandas DataFrame
    census = cell_census.open_soma(census_version="latest")

    # use subset of columns for speed
    obs_df = (
        census["census_data"][organism]
        .obs.read(column_names=["soma_joinid", "cell_type", "tissue"])
        .concat()
        .to_pandas()
    )
    assert len(obs_df)
    del obs_df

    var_df = census["census_data"][organism].ms["RNA"].var.read().concat().to_pandas()
    assert len(var_df)
    del var_df


def table_iter_is_ok(tbl_iter: collections.abc.Iterator[pa.Table], stop_after: Optional[int] = 2) -> None:
    assert isinstance(tbl_iter, collections.abc.Iterator)
    for n, tbl in enumerate(tbl_iter):
        # keep things speedy by quitting early if stop_after specified
        if stop_after is not None and n > stop_after:
            break
        assert isinstance(tbl, pa.Table)
        assert len(tbl)

    return True


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
def test_incremental_read(organism: str) -> None:
    # Verify that obs, var and X[raw] can be read incrementally, i.e., in chunks

    # Note: queries are reduced in scope (i.e., not all column_names or not all chunks read)
    # to speed up the tests.

    # open census with a small (default) TileDB buffer size, which both reduces
    # test memory use, and makes it run faster
    version = cell_census.get_census_version_description("latest")
    s3_region = version["soma"].get("s3_region")
    context = soma.options.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({"vfs.s3.region": s3_region}))
    with cell_census.open_soma(census_version="latest", context=context) as census:
        assert table_iter_is_ok(census["census_data"][organism].obs.read(column_names=["soma_joinid", "tissue"]))
        assert table_iter_is_ok(
            census["census_data"][organism].ms["RNA"].var.read(column_names=["soma_joinid", "feature_id"])
        )
        assert table_iter_is_ok(census["census_data"][organism].ms["RNA"].X["raw"].read().tables())


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
def test_incremental_query_quick(organism: str):
    # use default TileDB configuration
    with cell_census.open_soma(census_version="latest") as census:
        with census["census_data"][organism].axis_query(
            measurement_name="RNA", obs_query=soma.AxisQuery(value_filter="tissue_general == 'tongue'")
        ) as query:
            assert table_iter_is_ok(query.obs())
            assert table_iter_is_ok(query.var())
            assert table_iter_is_ok(query.X("raw").tables())


@pytest.mark.live_corpus
@pytest.mark.expensive
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
def test_incremental_query_full(organism: str):
    """Full (expensive) incremental read."""
    # use default TileDB configuration
    with cell_census.open_soma(census_version="latest") as census:
        with census["census_data"][organism].axis_query(
            measurement_name="RNA", obs_query=soma.AxisQuery(value_filter="tissue == 'brain'")
        ) as query:
            assert table_iter_is_ok(query.obs(), stop_after=None)
            assert table_iter_is_ok(query.var(), stop_after=None)
            assert table_iter_is_ok(query.X("raw").tables(), stop_after=None)


@pytest.mark.live_corpus
@pytest.mark.expensive
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
@pytest.mark.parametrize(
    "obs_value_filter",
    [
        "cell_type == 'neuron'",  # very common cell type
        "tissue == 'brain'",  # very common tissue
        None,  # whole enchilada
    ],
)
def test_get_anndata(organism: str, obs_value_filter: str) -> None:
    """Full (expensive) query and read into AnnData"""
    with cell_census.open_soma(census_version="latest") as census:
        ad = cell_census.get_anndata(census, organism, obs_value_filter=obs_value_filter)
        assert ad is not None
