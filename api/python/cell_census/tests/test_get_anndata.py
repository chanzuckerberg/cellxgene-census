import pytest
import tiledbsoma as soma

import cell_census


@pytest.fixture
def census() -> soma.Collection:
    return cell_census.open_soma(census_version="latest")


@pytest.mark.live_corpus
def test_get_anndata_value_filter(census: soma.Collection) -> None:
    with census:
        ad = cell_census.get_anndata(
            census,
            organism="Mus musculus",
            obs_value_filter="tissue_general == 'vasculature'",
            var_value_filter="feature_name in ['Gm53058', '0610010K14Rik']",
            column_names={
                "obs": ["soma_joinid", "cell_type", "tissue", "tissue_general", "assay"],
                "var": ["soma_joinid", "feature_id", "feature_name", "feature_length"],
            },
        )

    assert ad is not None
    assert ad.n_vars == 2
    assert ad.n_obs > 0
    assert (ad.obs.tissue_general == "vasculature").all()
    assert set(ad.var.feature_name) == {"Gm53058", "0610010K14Rik"}


@pytest.mark.live_corpus
def test_get_anndata_coords(census: soma.Collection) -> None:
    with census:
        ad = cell_census.get_anndata(census, organism="Mus musculus", obs_coords=slice(1000), var_coords=slice(2000))

    assert ad is not None
    assert ad.n_vars == 2001
    assert ad.n_obs == 1001


@pytest.mark.live_corpus
def test_get_anndata_allows_missing_obs_or_var_filter(census: soma.Collection) -> None:
    # This test is slightly sensitive to the live data, in that it assumes the
    # existance of certain cell tissue labels and gene feature ids.
    with census:
        mouse = census["census_data"]["mus_musculus"]

        adata = cell_census.get_anndata(census, organism="Mus musculus", obs_value_filter="tissue == 'aorta'")
        assert adata.n_obs == len(
            mouse.obs.read(value_filter="tissue == 'aorta'", column_names=["soma_joinid"]).concat()
        )
        assert adata.n_vars == len(mouse.ms["RNA"].var.read(column_names=["soma_joinid"]).concat())

        adata = cell_census.get_anndata(
            census,
            organism="Mus musculus",
            obs_coords=slice(10000),
            var_value_filter="feature_id == 'ENSMUSG00000069581'",
        )
        assert adata.n_obs == 10001
        assert adata.n_vars == 1
