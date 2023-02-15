import pytest
import scipy.sparse

import cell_census


@pytest.mark.live_corpus
def test_get_experiment() -> None:
    with cell_census.open_soma(census_version="latest") as census:
        mouse_uri = census["census_data"]["mus_musculus"].uri
        human_uri = census["census_data"]["homo_sapiens"].uri

        assert cell_census.get_experiment(census, "mus musculus").uri == mouse_uri
        assert cell_census.get_experiment(census, "Mus musculus").uri == mouse_uri
        assert cell_census.get_experiment(census, "mus_musculus").uri == mouse_uri

        assert cell_census.get_experiment(census, "homo sapiens").uri == human_uri
        assert cell_census.get_experiment(census, "Homo sapiens").uri == human_uri
        assert cell_census.get_experiment(census, "homo_sapiens").uri == human_uri

    with pytest.raises(ValueError):
        cell_census.get_experiment(census, "no such critter")


@pytest.mark.live_corpus
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
def test_get_presence_matrix(organism: str) -> None:
    census = cell_census.open_soma(census_version="latest")

    census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()

    pm = cell_census.get_presence_matrix(census, organism)
    assert isinstance(pm, scipy.sparse.csr_matrix)
    assert pm.shape[0] == len(census_datasets)
    assert pm.shape[1] == len(
        census["census_data"][organism].ms["RNA"].var.read(column_names=["soma_joinid"]).concat().to_pandas()
    )
