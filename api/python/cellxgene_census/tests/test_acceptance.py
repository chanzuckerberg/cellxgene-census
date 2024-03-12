"""
Acceptance tests for the Census.

NOTE: those marked `expensive` are not run in the CI as they are, well, expensive...

Several of them will not run to completion except on VERY large hosts.

Intended use:  periodically do a manual run, including the expensive tests, on an
appropriately large host.

See README.md for historical data.
"""

from typing import Any, Dict, Iterator, Optional, Tuple

import pyarrow as pa
import pytest
import tiledbsoma as soma

import cellxgene_census
from cellxgene_census._open import DEFAULT_TILEDB_CONFIGURATION


def make_context(census_version: str, config: Optional[Dict[str, Any]] = None) -> soma.SOMATileDBContext:
    config = config or {}
    version = cellxgene_census.get_census_version_description(census_version)
    s3_region = version["soma"].get("s3_region", "us-west-2")
    config.update({"vfs.s3.region": s3_region})
    config.update({"vfs.s3.no_sign_request": "true"})
    return soma.options.SOMATileDBContext().replace(**{"tiledb_config": config})


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
def test_load_axes(organism: str) -> None:
    """Verify axes can be loaded into a Pandas DataFrame"""
    census = cellxgene_census.open_soma(census_version="latest")

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
@pytest.mark.parametrize(
    ("stop_after", "ctx_config"),
    [
        pytest.param(2, None),
        pytest.param(None, DEFAULT_TILEDB_CONFIGURATION, marks=pytest.mark.expensive),
    ],
)
def test_incremental_read_obs(organism: str, stop_after: Optional[int], ctx_config: Optional[Dict[str, Any]]) -> None:
    """Verify that obs, var and X[raw] can be read incrementally, i.e., in chunks"""

    # ctx_config=None open census with a small (default) TileDB buffer size, which reduces
    # memory use, and makes it feasible to run in a GHA.
    ctx_config = ctx_config or {}
    context = make_context("latest", ctx_config)
    with cellxgene_census.open_soma(census_version="latest", context=context) as census:
        assert table_iter_is_ok(
            census["census_data"][organism].obs.read(column_names=["soma_joinid", "tissue"]),
            stop_after=stop_after,
        )


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
@pytest.mark.parametrize(
    ("stop_after", "ctx_config"),
    [
        pytest.param(2, None),
        pytest.param(None, DEFAULT_TILEDB_CONFIGURATION, marks=pytest.mark.expensive),
    ],
)
def test_incremental_read_var(organism: str, stop_after: Optional[int], ctx_config: Optional[Dict[str, Any]]) -> None:
    """Verify that var can be read incrementally, i.e., in chunks"""

    # ctx_config=None open census with a small (default) TileDB buffer size, which reduces
    # memory use, and makes it feasible to run in a GHA.
    ctx_config = ctx_config or {}
    context = make_context("latest", ctx_config)
    with cellxgene_census.open_soma(census_version="latest", context=context) as census:
        assert table_iter_is_ok(
            census["census_data"][organism].ms["RNA"].var.read(column_names=["soma_joinid", "feature_id"]),
            stop_after=stop_after,
        )


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
@pytest.mark.parametrize(
    ("stop_after", "ctx_config", "coords"),
    [
        pytest.param(2, None, (None, None)),
        pytest.param(
            None,
            DEFAULT_TILEDB_CONFIGURATION,
            (slice(0, 500_000), slice(None)),
            marks=pytest.mark.expensive,
        ),
        pytest.param(
            None,
            {"soma.init_buffer_bytes": 4 * 1024**3},
            (slice(0, 500_000), slice(None)),
            marks=pytest.mark.expensive,
        ),
        pytest.param(
            None,
            {"soma.init_buffer_bytes": 4 * 1024**3},
            (slice(0, 500_000), slice(0, 1_000)),
            marks=pytest.mark.expensive,
        ),
    ],
)
def test_incremental_read_X(
    organism: str,
    stop_after: Optional[int],
    ctx_config: Optional[Dict[str, Any]],
    coords: Optional[Tuple[slice, slice]],
) -> None:
    """Verify that obs, var and X[raw] can be read incrementally, i.e., in chunks"""

    ctx_config = ctx_config or {}
    context = make_context("latest", ctx_config)
    with cellxgene_census.open_soma(census_version="latest", context=context) as census:
        assert table_iter_is_ok(
            census["census_data"][organism].ms["RNA"].X["raw"].read(coords=coords).tables(),
            stop_after=stop_after,
        )


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
@pytest.mark.parametrize(
    "obs_value_filter",
    ["tissue=='aorta'", pytest.param("tissue=='brain'", marks=pytest.mark.expensive)],
)
@pytest.mark.parametrize("stop_after", [2, pytest.param(None, marks=pytest.mark.expensive)])
def test_incremental_query(organism: str, obs_value_filter: str, stop_after: Optional[int]) -> None:
    """Verify incremental read of query result."""
    # use default TileDB configuration
    with cellxgene_census.open_soma(census_version="latest") as census:
        with census["census_data"][organism].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=obs_value_filter),
        ) as query:
            assert table_iter_is_ok(query.obs(), stop_after=stop_after)
            assert table_iter_is_ok(query.var(), stop_after=stop_after)
            assert table_iter_is_ok(query.X("raw").tables(), stop_after=stop_after)


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
@pytest.mark.parametrize(
    ("obs_value_filter", "obs_coords", "ctx_config"),
    [
        # small query, should be runable in CI
        pytest.param("tissue=='aorta'", None, DEFAULT_TILEDB_CONFIGURATION),
        # 10K cells, also small enough to run in CI
        pytest.param(None, slice(0, 10_000), DEFAULT_TILEDB_CONFIGURATION, id="First 10K cells"),
        # 100K cells, standard buffer size
        pytest.param(
            None,
            slice(0, 100_000),
            DEFAULT_TILEDB_CONFIGURATION,
            marks=pytest.mark.expensive,
            id="First 100K cells",
        ),
        # 250K cells, standard buffer size
        pytest.param(
            None,
            slice(0, 250_000),
            DEFAULT_TILEDB_CONFIGURATION,
            marks=pytest.mark.expensive,
            id="First 250K cells",
        ),
        # 500K cells, standard buffer size
        pytest.param(
            None,
            slice(0, 500_000),
            DEFAULT_TILEDB_CONFIGURATION,
            marks=pytest.mark.expensive,
            id="First 500K cells",
        ),
        # 750K cells, standard buffer size
        pytest.param(
            None,
            slice(0, 750_000),
            DEFAULT_TILEDB_CONFIGURATION,
            marks=pytest.mark.expensive,
            id="First 750K cells",
        ),
        # 1M cells, standard buffer size
        pytest.param(
            None,
            slice(0, 1_000_000),
            DEFAULT_TILEDB_CONFIGURATION,
            marks=pytest.mark.expensive,
            id="First 1M cells",
        ),
        # very common tissue, with standard buffer size
        pytest.param(
            "tissue_general=='brain'",
            slice(0, 1_000_000),
            DEFAULT_TILEDB_CONFIGURATION,
            marks=pytest.mark.expensive,
        ),
        # very common cell type, with standard buffer size
        pytest.param(
            "cell_type=='neuron'",
            slice(0, 1_000_000),
            DEFAULT_TILEDB_CONFIGURATION,
            marks=pytest.mark.expensive,
        ),
        # many primary cells, with big buffer size
        pytest.param(
            "is_primary_data==True",
            slice(0, 1_000_000),
            {"soma.init_buffer_bytes": 4 * 1024**3},
            marks=pytest.mark.expensive,
        ),
        #
        # a large enchilada, if not the whole thing, with a big buffer size
        pytest.param(
            None,
            slice(0, 1_000_000),
            {"soma.init_buffer_bytes": 4 * 1024**3},
            marks=pytest.mark.expensive,
        ),
    ],
)
def test_get_anndata(
    organism: str,
    obs_value_filter: Optional[str],
    obs_coords: Optional[slice],
    ctx_config: Optional[Dict[str, Any]],
) -> None:
    """Verify query and read into AnnData"""
    ctx_config = ctx_config or {}
    context = make_context("latest", ctx_config)
    with cellxgene_census.open_soma(census_version="latest", context=context) as census:
        ad = cellxgene_census.get_anndata(census, organism, obs_value_filter=obs_value_filter, obs_coords=obs_coords)
        assert ad is not None

        # sanity checks
        with census["census_data"][organism].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=obs_value_filter, coords=(obs_coords,)),
        ) as query:
            assert ad.n_obs == query.n_obs
            assert ad.n_vars == query.n_vars
