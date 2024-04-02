import pytest
import requests_mock as rm

from cellxgene_census.experimental import (
    get_all_available_embeddings,
    get_all_census_versions_with_embedding,
    get_embedding_metadata_by_name,
)
from cellxgene_census.experimental._embedding import CELL_CENSUS_EMBEDDINGS_MANIFEST_URL


def test_get_embedding_metadata_by_name(requests_mock: rm.Mocker) -> None:
    mock_embeddings = {
        "embedding-id-1": {
            "id": "embedding-id-1",
            "embedding_name": "emb_1",
            "title": "Embedding 1",
            "description": "First embedding",
            "experiment_name": "homo_sapiens",
            "data_type": "obs_embedding",
            "census_version": "2023-12-15",
            "submission_date": "2023-11-15",
        },
        "embedding-id-2": {
            "id": "embedding-id-2",
            "embedding_name": "emb_1",
            "title": "Embedding 2",
            "description": "Second embedding",
            "experiment_name": "homo_sapiens",
            "data_type": "obs_embedding",
            "census_version": "2023-12-15",
            "submission_date": "2023-12-31",
        },
        "embedding-id-3": {
            "id": "embedding-id-3",
            "embedding_name": "emb_3",
            "title": "Embedding 3",
            "description": "Third embedding",
            "experiment_name": "homo_sapiens",
            "data_type": "obs_embedding",
            "census_version": "2023-12-15",
            "submission_date": "2023-11-15",
        },
    }
    requests_mock.real_http = True
    requests_mock.get(CELL_CENSUS_EMBEDDINGS_MANIFEST_URL, json=mock_embeddings)

    embedding = get_embedding_metadata_by_name(
        "emb_1", organism="homo_sapiens", census_version="2023-12-15", embedding_type="obs_embedding"
    )
    assert embedding is not None
    assert embedding["id"] == "embedding-id-2"  # most recent version
    assert embedding == mock_embeddings["embedding-id-2"]

    embedding = get_embedding_metadata_by_name(
        "emb_3", organism="homo_sapiens", census_version="2023-12-15", embedding_type="obs_embedding"
    )
    assert embedding is not None
    assert embedding["id"] == "embedding-id-3"
    assert embedding == mock_embeddings["embedding-id-3"]

    with pytest.raises(ValueError):
        get_embedding_metadata_by_name(
            "emb_2", organism="homo_sapiens", census_version="2023-12-15", embedding_type="obs_embedding"
        )
        get_embedding_metadata_by_name(
            "emb_1", organism="mus_musculus", census_version="2023-12-15", embedding_type="obs_embedding"
        )
        get_embedding_metadata_by_name(
            "emb_1", organism="homo_sapiens", census_version="2023-10-15", embedding_type="obs_embedding"
        )
        get_embedding_metadata_by_name(
            "emb_1", organism="mus_musculus", census_version="2023-12-15", embedding_type="var_embedding"
        )


def test_get_all_available_embeddings(requests_mock: rm.Mocker) -> None:
    mock_embeddings = {
        "embedding-id-1": {
            "id": "embedding-id-1",
            "embedding_name": "emb_1",
            "title": "Embedding 1",
            "description": "First embedding",
            "experiment_name": "homo_sapiens",
            "measurement_name": "RNA",
            "n_embeddings": 1000,
            "n_features": 200,
            "data_type": "obs_embedding",
            "census_version": "2023-12-15",
        },
        "embedding-id-2": {
            "id": "embedding-id-2",
            "embedding_name": "emb_2",
            "title": "Embedding 2",
            "description": "Second embedding",
            "experiment_name": "homo_sapiens",
            "measurement_name": "RNA",
            "n_embeddings": 1000,
            "n_features": 200,
            "data_type": "obs_embedding",
            "census_version": "2023-12-15",
        },
    }
    requests_mock.real_http = True
    requests_mock.get(CELL_CENSUS_EMBEDDINGS_MANIFEST_URL, json=mock_embeddings)

    embeddings = get_all_available_embeddings("2023-12-15")
    assert embeddings is not None
    assert len(embeddings) == 2

    # Query for a non existing version of the Census
    embeddings = get_all_available_embeddings("2024-12-15")
    assert len(embeddings) == 0


def test_get_all_census_versions_with_embedding(requests_mock: rm.Mocker) -> None:
    mock_embeddings = {
        "embedding-id-1": {
            "id": "embedding-id-1",
            "embedding_name": "emb_1",
            "title": "Embedding 1",
            "description": "First embedding",
            "experiment_name": "homo_sapiens",
            "data_type": "obs_embedding",
            "census_version": "2023-12-15",
        },
        "embedding-id-2": {
            "id": "embedding-id-2",
            "embedding_name": "emb_1",
            "title": "Embedding 2",
            "description": "Second embedding",
            "experiment_name": "homo_sapiens",
            "data_type": "obs_embedding",
            "census_version": "2023-12-15",
        },
        "embedding-id-3": {
            "id": "embedding-id-3",
            "embedding_name": "emb_1",
            "title": "Embedding 3",
            "description": "Third embedding",
            "experiment_name": "mus_musculus",
            "data_type": "obs_embedding",
            "census_version": "2023-12-15",
        },
        "embedding-id-4": {
            "id": "embedding-id-4",
            "embedding_name": "emb_1",
            "title": "Embedding 4",
            "description": "Fourth embedding",
            "experiment_name": "mus_musculus",
            "data_type": "obs_embedding",
            "census_version": "2024-01-01",
        },
        "embedding-id-5": {
            "id": "embedding-id-5",
            "embedding_name": "emb_2",
            "title": "Embedding 5",
            "description": "Fifth embedding",
            "experiment_name": "mus_musculus",
            "data_type": "var_embedding",
            "census_version": "2023-12-15",
        },
    }
    requests_mock.real_http = True
    requests_mock.get(CELL_CENSUS_EMBEDDINGS_MANIFEST_URL, json=mock_embeddings)

    versions = get_all_census_versions_with_embedding("emb_1", organism="homo_sapiens", embedding_type="obs_embedding")
    assert versions == ["2023-12-15"]

    versions = get_all_census_versions_with_embedding("emb_1", organism="mus_musculus", embedding_type="obs_embedding")
    assert versions == ["2023-12-15", "2024-01-01"]

    versions = get_all_census_versions_with_embedding("emb_1", organism="mus_musculus", embedding_type="var_embedding")
    assert versions == []

    versions = get_all_census_versions_with_embedding("emb_2", organism="mus_musculus", embedding_type="var_embedding")
    assert versions == ["2023-12-15"]
