from typing import List

import numpy as np
import pytest
import tiledbsoma as soma

import cellxgene_census


@pytest.fixture
def census() -> soma.Collection:
    return cellxgene_census.open_soma(census_version="latest")


@pytest.fixture
def lts_census() -> soma.Collection:
    return cellxgene_census.open_soma(census_version="stable")


@pytest.mark.live_corpus
def test_get_anndata_value_filter(census: soma.Collection) -> None:
    with census:
        ad = cellxgene_census.get_anndata(
            census,
            organism="Mus musculus",
            obs_value_filter="tissue_general == 'vasculature'",
            var_value_filter="feature_name in ['Gm53058', '0610010K14Rik']",
            column_names={
                "obs": [
                    "soma_joinid",
                    "cell_type",
                    "tissue",
                    "tissue_general",
                    "assay",
                ],
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
        ad = cellxgene_census.get_anndata(
            census,
            organism="Mus musculus",
            obs_coords=slice(1000),
            var_coords=slice(2000),
        )

    assert ad is not None
    assert ad.n_vars == 2001
    assert ad.n_obs == 1001


@pytest.mark.live_corpus
def test_get_anndata_allows_missing_obs_or_var_filter(census: soma.Collection) -> None:
    # This test is slightly sensitive to the live data, in that it assumes the
    # existance of certain cell tissue labels and gene feature ids.
    with census:
        mouse = census["census_data"]["mus_musculus"]

        adata = cellxgene_census.get_anndata(census, organism="Mus musculus", obs_value_filter="tissue == 'aorta'")
        assert adata.n_obs == len(
            mouse.obs.read(value_filter="tissue == 'aorta'", column_names=["soma_joinid"]).concat()
        )
        assert adata.n_vars == len(mouse.ms["RNA"].var.read(column_names=["soma_joinid"]).concat())

        adata = cellxgene_census.get_anndata(
            census,
            organism="Mus musculus",
            obs_coords=slice(10000),
            var_value_filter="feature_id == 'ENSMUSG00000069581'",
        )
        assert adata.n_obs == 10001
        assert adata.n_vars == 1


@pytest.mark.live_corpus
@pytest.mark.parametrize("layer", ["raw", "normalized"])
def test_get_anndata_x_layer(census: soma.Collection, layer: str) -> None:
    with census:
        ad = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            X_name=layer,
            obs_coords=slice(100),
            var_coords=slice(200),
        )

    assert ad.X.shape == (101, 201)
    assert len(ad.layers) == 0


@pytest.mark.live_corpus
@pytest.mark.parametrize("layers", [["raw", "normalized"], ["normalized", "raw"]])
def test_get_anndata_two_layers(census: soma.Collection, layers: List[str]) -> None:
    with census:
        ad_primary_layer_in_X = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            X_name=layers[0],
            obs_coords=slice(100),
            var_coords=slice(200),
        )

        ad_secondary_layer_in_X = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            X_name=layers[1],
            obs_coords=slice(100),
            var_coords=slice(200),
        )

        ad_multiple_layers = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            X_name=layers[0],
            X_layers=[layers[1]],
            obs_coords=slice(100),
            var_coords=slice(200),
        )

    assert layers[1] in ad_multiple_layers.layers
    assert ad_multiple_layers.X.shape == (101, 201)
    assert ad_multiple_layers.layers[layers[1]].shape == (101, 201)

    # Assert that matrices of multilayer anndata are equal to one-layer-at-time anndatas
    assert np.array_equal(ad_multiple_layers.X.data, ad_primary_layer_in_X.X.data)
    assert np.array_equal(ad_multiple_layers.layers[layers[1]].data, ad_secondary_layer_in_X.X.data)


@pytest.mark.live_corpus
def test_get_anndata_wrong_layer_names(census: soma.Collection) -> None:
    with census:
        with pytest.raises(ValueError) as raise_info:
            cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                X_name="this_layer_name_is_bad",
                obs_coords=slice(100),
                var_coords=slice(200),
            )

            assert raise_info.value.args[0] == "Unknown X layer name"

        with pytest.raises(ValueError) as raise_info:
            cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                X_name="raw",
                X_layers=["this_layer_name_is_bad"],
                obs_coords=slice(100),
                var_coords=slice(200),
            )

            assert raise_info.value.args[0] == "Unknown X layer name"


@pytest.mark.live_corpus
@pytest.mark.parametrize("obsm_layer", ["scvi", "geneformer"])
def test_get_anndata_obsm_one_layer(lts_census: soma.Collection, obsm_layer: str) -> None:
    # NOTE: this test will break after next LTS release (>2023-12-15), since scvi and geneformer
    # won't be distributed as part of `obsm_layers` anymore. Delete this test when it happens.
    with lts_census:
        ad = cellxgene_census.get_anndata(
            lts_census,
            organism="Homo sapiens",
            X_name="raw",
            obs_coords=slice(100),
            var_coords=slice(200),
            obsm_layers=[obsm_layer],
        )

    assert len(ad.obsm.keys()) == 1
    assert obsm_layer in ad.obsm.keys()
    assert ad.obsm[obsm_layer].shape[0] == 101


@pytest.mark.live_corpus
@pytest.mark.parametrize("obsm_layers", [["scvi", "geneformer"]])
def test_get_anndata_obsm_two_layers(lts_census: soma.Collection, obsm_layers: List[str]) -> None:
    # NOTE: this test will break after next LTS release (>2023-12-15), since scvi and geneformer
    # won't be distributed as part of `obsm_layers` anymore. Delete this test when it happens.
    with lts_census:
        ad = cellxgene_census.get_anndata(
            lts_census,
            organism="Homo sapiens",
            X_name="raw",
            obs_coords=slice(100),
            var_coords=slice(200),
            obsm_layers=obsm_layers,
        )

    assert len(ad.obsm.keys()) == 2
    for obsm_layer in obsm_layers:
        assert obsm_layer in ad.obsm.keys()
        assert ad.obsm[obsm_layer].shape[0] == 101


@pytest.mark.live_corpus
@pytest.mark.parametrize("obs_embeddings", [["scvi", "geneformer", "uce"]])
def test_get_anndata_obs_embeddings(lts_census: soma.Collection, obs_embeddings: List[str]) -> None:
    # NOTE: when the next LTS gets released (>2023-12-15), embeddings may or may not be available,
    # so this test could require adjustments.

    with lts_census:
        ad = cellxgene_census.get_anndata(
            lts_census,
            organism="Homo sapiens",
            X_name="raw",
            obs_coords=slice(100),
            var_coords=slice(200),
            obs_embeddings=obs_embeddings,
        )

    assert len(ad.obsm.keys()) == 3
    assert len(ad.varm.keys()) == 0
    for obsm_layer in obs_embeddings:
        assert obsm_layer in ad.obsm.keys()
        assert ad.obsm[obsm_layer].shape[0] == 101


@pytest.mark.live_corpus
@pytest.mark.parametrize("var_embeddings", [["nmf"]])
def test_get_anndata_var_embeddings(lts_census: soma.Collection, var_embeddings: List[str]) -> None:
    # NOTE: when the next LTS gets released (>2023-12-15), embeddings may or may not be available,
    # so this test could require adjustments.

    with lts_census:
        ad = cellxgene_census.get_anndata(
            lts_census,
            organism="Homo sapiens",
            X_name="raw",
            obs_coords=slice(100),
            var_coords=slice(200),
            var_embeddings=var_embeddings,
        )

    assert len(ad.obsm.keys()) == 0
    assert len(ad.varm.keys()) == 1
    for varm_layers in var_embeddings:
        assert varm_layers in ad.varm.keys()
        assert ad.varm[varm_layers].shape[0] == 201


@pytest.mark.live_corpus
def test_get_anndata_obsm_layers_and_add_obs_embedding_fails(lts_census: soma.Collection) -> None:
    """Fails if both `obsm_layers` and `obs_embeddings` are specified."""
    with lts_census:
        with pytest.raises(ValueError):
            cellxgene_census.get_anndata(
                lts_census,
                organism="Homo sapiens",
                X_name="raw",
                obs_coords=slice(100),
                var_coords=slice(200),
                obsm_layers=["scvi"],
                obs_embeddings=["scvi"],
            )
