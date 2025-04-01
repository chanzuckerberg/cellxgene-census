import pytest
import scipy.sparse
import tiledbsoma as soma

import cellxgene_census
from cellxgene_census._experiment import _get_experiment


@pytest.mark.live_corpus
def test_get_experiment(census: soma.Collection) -> None:
    mouse_uri = census["census_data"]["mus_musculus"].uri
    human_uri = census["census_data"]["homo_sapiens"].uri

    assert _get_experiment(census, "mus musculus").uri == mouse_uri
    assert _get_experiment(census, "Mus musculus").uri == mouse_uri
    assert _get_experiment(census, "mus_musculus").uri == mouse_uri

    assert _get_experiment(census, "homo sapiens").uri == human_uri
    assert _get_experiment(census, "Homo sapiens").uri == human_uri
    assert _get_experiment(census, "homo_sapiens").uri == human_uri

    with pytest.raises(ValueError):
        _get_experiment(census, "no such critter")


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
@pytest.mark.parametrize("modality", ["census_data", "census_spatial_sequencing"])
def test_get_presence_matrix(organism: str, census: soma.Collection, modality: str) -> None:
    census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()

    pm = cellxgene_census.get_presence_matrix(census, organism, modality=modality)
    assert isinstance(pm, scipy.sparse.csr_matrix)
    assert pm.shape[0] == len(census_datasets)
    assert pm.shape[1] == len(
        census[modality][organism].ms["RNA"].var.read(column_names=["soma_joinid"]).concat().to_pandas()
    )
