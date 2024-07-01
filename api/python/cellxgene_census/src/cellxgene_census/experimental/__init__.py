"""Experimental API for the CELLxGENE Discover Census."""

from ._embedding import (
    get_all_available_embeddings,
    get_all_census_versions_with_embedding,
    get_embedding,
    get_embedding_metadata,
    get_embedding_metadata_by_name,
)

__all__ = [
    "get_embedding",
    "get_embedding_metadata",
    "get_embedding_metadata_by_name",
    "get_all_available_embeddings",
    "get_all_census_versions_with_embedding",
]
