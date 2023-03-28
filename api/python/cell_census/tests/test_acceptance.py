"""
Acceptance tests for the Census.

NOTE: those marked `expensive` are not run in the CI as they are, well, expensive...

Several of them will not run to completion except on VERY large hosts.

Intended use:  periodically do a manual run, including the expensive tests, on an
appropriately large host.

See README.md for historical data.
"""
from typing import Iterator, Optional

import pyarrow as pa
import pytest
import tiledb
import tiledbsoma as soma

import cell_census


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
def test_load_axes(organism: str) -> None:
    """Verify axes can be loaded into a Pandas DataFrame"""
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


def table_iter_is_ok(tbl_iter: Iterator[pa.Table], stop_after: Optional[int] = 2) -> bool:
    """
    Utility that verifies that the value is an iterator of pa.Table.

    Will only call __next__ as many times as the `stop_after` param specifies,
    or will read until end of iteration of it is None.
    """
    assert isinstance(tbl_iter, Iterator)
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
    """Verify that obs, var and X[raw] can be read incrementally, i.e., in chunks"""
    with cell_census.open_soma(census_version="latest") as census:
        assert table_iter_is_ok(census["census_data"][organism].obs.read(column_names=["soma_joinid", "tissue"]))
        assert table_iter_is_ok(
            census["census_data"][organism].ms["RNA"].var.read(column_names=["soma_joinid", "feature_id"])
        )
        assert table_iter_is_ok(census["census_data"][organism].ms["RNA"].X["raw"].read().tables())


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
@pytest.mark.parametrize(
    "obs_value_filter", ["tissue=='aorta'", pytest.param("tissue=='brain'", marks=pytest.mark.expensive)]
)
@pytest.mark.parametrize("stop_after", [2, pytest.param(None, marks=pytest.mark.expensive)])
def test_incremental_query(organism: str, obs_value_filter: str, stop_after: Optional[int]) -> None:
    """Verify incremental read of query result."""
    # use default TileDB configuration
    with cell_census.open_soma(census_version="latest") as census:
        with census["census_data"][organism].axis_query(
            measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
        ) as query:
            assert table_iter_is_ok(query.obs(), stop_after=stop_after)
            assert table_iter_is_ok(query.var(), stop_after=stop_after)
            assert table_iter_is_ok(query.X("raw").tables(), stop_after=stop_after)


@pytest.mark.live_corpus
@pytest.mark.expensive
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
@pytest.mark.parametrize(
    "obs_value_filter",
    [
        "tissue == 'aorta'",
        pytest.param("cell_type == 'neuron'", marks=pytest.mark.expensive),  # very common cell type
        pytest.param("tissue == 'brain'", marks=pytest.mark.expensive),  # very common tissue
        pytest.param(None, marks=pytest.mark.expensive),  # whole enchilada
    ],
)
def test_get_anndata(organism: str, obs_value_filter: str) -> None:
    """Verify query and read into AnnData"""
    with cell_census.open_soma(census_version="latest") as census:
        ad = cell_census.get_anndata(census, organism, obs_value_filter=obs_value_filter)
        assert ad is not None
