"""
Compatibility tests between the installed verison of cellxgene-census and
a named LTS release. Primarilly intended to be driven by a periodic GHA.

Where there are known and accepted incompatibilities, use `pytest.skipif`
"""

import pytest

import cellxgene_census


@pytest.mark.lts_compat_check
def test_open(census_version: str) -> None:
    print("census-version", census_version)
    with cellxgene_census.open_soma(census_version=census_version) as census:
        assert repr(census)
