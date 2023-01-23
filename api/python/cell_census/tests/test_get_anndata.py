import pytest
import tiledbsoma as soma

import cell_census


@pytest.fixture
def census() -> soma.Collection:
    return cell_census.open_soma(census_version="latest")


@pytest.mark.live_corpus
def test_get_anndata(census: soma.Collection) -> None:

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
def test_get_anndata_allows_missing_obs_or_var_filter(census: soma.Collection) -> None:
    # TODO: test with a small, local census test fixture, for performance; reinstate commented-out test, below

    adata = cell_census.get_anndata(census, organism="Homo sapiens", obs_value_filter="tissue == 'tongue'")
    assert adata.obs.shape[0] == 372

    # adata = cell_census.get_anndata(
    #         census,
    #         organism="Homo sapiens",
    #         var_value_filter=f"feature_id == 'TP53'"
    # )
    # assert adata.var.shape[0] == 1
