"""
Compatibility tests between the installed verison of cellxgene-census and
a named LTS release. Primarilly intended to be driven by a periodic GHA.

Where there are known and accepted incompatibilities, use `pytest.skip`
to codify them.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator, Literal, Sequence, Union, get_args

import pyarrow as pa
import pytest
import tiledbsoma as soma

import cellxgene_census

SOMATypeNames = Literal[
    "SOMACollection",
    "SOMAExperiment",
    "SOMAMeasurement",
    "SOMADataFrame",
    "SOMASparseNDArray",
    "SOMADenseNDArray",
]
CollectionTypeNames = ["SOMACollection", "SOMAExperiment", "SOMAMeasurement"]

SOMATypes = Union[
    soma.Collection,
    soma.DataFrame,
    soma.SparseNDArray,
    soma.DenseNDArray,
    soma.Experiment,
    soma.Measurement,
]


def walk_census(
    census: soma.Collection, filter_types: Sequence[SOMATypeNames] | None = None
) -> Iterator[tuple[str, SOMATypes]]:
    assert census.soma_type == "SOMACollection"
    filter_types = filter_types or get_args(SOMATypeNames)
    items_to_check = deque([("census", census)])
    while items_to_check:
        key, val = items_to_check.popleft()
        if val.soma_type in CollectionTypeNames:
            items_to_check.extend(val.items())

        if val.soma_type not in filter_types:
            continue

        yield key, val


@pytest.mark.lts_compat_check
def test_open(census_version: str) -> None:
    """
    Verify we can open and walk the collections, get metadata and read schema on non-collections
    """

    with cellxgene_census.open_soma(census_version=census_version) as census:
        for name, item in walk_census(census):
            assert name
            assert list(item.metadata)
            if item.soma_type not in CollectionTypeNames:
                assert isinstance(item.schema, pa.Schema)


@pytest.mark.lts_compat_check
def test_read_dataframe(census_version: str) -> None:
    """
    Verify we can read at least one row of dataframes
    """
    with cellxgene_census.open_soma(census_version=census_version) as census:
        for name, sdf in walk_census(census, filter_types=["SOMADataFrame"]):
            assert name
            # the Census should have no zero-length DataFrames
            assert len(sdf)
            df = sdf.read(coords=([0],)).concat().to_pandas()
            assert len(df) == 1
            assert df.shape == (1, len(sdf.keys()))


@pytest.mark.lts_compat_check
def test_read_arrays(census_version: str) -> None:
    """
    Verify we can read from NDArray
    """
    with cellxgene_census.open_soma(census_version=census_version) as census:
        for name, sarr in walk_census(census, filter_types=["SOMASparseNDArray", "SOMADenseNDArray"]):
            assert name
            assert isinstance(sarr.shape, tuple)

            # There are currently no Census schema versions using DenseNDArray

            assert sarr.soma_type == "SOMASparseNDArray"
            assert sarr.nnz
            tbl = sarr.read(coords=(slice(100),)).tables().concat()
            assert len(tbl) > 0
