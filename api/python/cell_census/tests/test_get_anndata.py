import cell_census


def test_get_anndata() -> None:
    census = cell_census.open_soma(census_version="latest")
    ad = cell_census.get_anndata(
        census,
        organism="Mus musculus",
        obs_query={"tissue_general": "vasculature"},
        var_query={"feature_name": ["Gm53058", "0610010K14Rik"]},
        column_names={
            "obs": ["cell_type", "tissue", "tissue_general", "assay"],
            "var": ["feature_id", "feature_name", "feature_length"],
        },
    )

    assert ad is not None
    assert ad.n_vars == 2
    assert ad.n_obs > 0
    assert (ad.obs.tissue_general == "vasculature").all()
    assert set(ad.var.feature_name) == {"Gm53058", "0610010K14Rik"}
