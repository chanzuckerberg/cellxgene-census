import pytest

import cell_census


@pytest.mark.live_corpus
def test_get_anndata() -> None:
    census = cell_census.open_soma(census_version="latest")
    ad = cell_census.get_anndata(
        census,
        organism="Mus musculus",
        obs_value_filter={"tissue_general": "vasculature"},
        var_value_filter={"feature_name": ["Gm53058", "0610010K14Rik"]},
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

    # Repeat the same query, but by joinid
    ad2 = cell_census.get_anndata(
        census,
        organism="Mus musculus",
        obs_joinids=ad.obs.soma_joinid.to_numpy(),
        var_joinids=ad.var.soma_joinid.to_numpy(),
        column_names={
            "obs": ad.obs.keys().to_list(),
            "var": ad.var.keys().to_list(),
        },
    )
    assert ad.obs.equals(ad2.obs)
    assert ad.var.equals(ad2.var)
    assert ad.X.shape == ad2.X.shape
    assert (ad.X != ad2.X).nnz == 0


@pytest.mark.live_corpus
def test_get_anndata_error_checks() -> None:
    census = cell_census.open_soma(census_version="latest")

    # verify that common query errors are caught

    with pytest.raises(ValueError):
        cell_census.get_anndata(
            census,
            organism="Mus musculus",
            obs_value_filter={"tissue_general": "vasculature"},
            obs_joinids=[0, 1],
            var_value_filter={"feature_name": ["Gm53058", "0610010K14Rik"]},
        )

    with pytest.raises(ValueError):
        cell_census.get_anndata(
            census,
            organism="Mus musculus",
            obs_value_filter={"tissue_general": "vasculature"},
            var_value_filter={"feature_name": ["Gm53058", "0610010K14Rik"]},
            var_joinids=[0, 1],
        )

    with pytest.raises(TypeError):
        cell_census.get_anndata(
            census,
            organism="Mus musculus",
            obs_value_filter={"tissue_general": {}},  # type: ignore
        )
