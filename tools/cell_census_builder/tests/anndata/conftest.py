from typing import List

import pytest

from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.tests.conftest import ORGANISMS, get_h5ad


@pytest.fixture
def datasets_with_mixed_feature_reference(assets_path: str) -> List[Dataset]:
    organism = ORGANISMS[0]
    dataset_id = "an_id"
    datasets = []
    h5ad = get_h5ad(organism)
    # Modify a row of `var` so that the dataset has a mixed feature_reference
    h5ad.var.at["homo_sapiens_c", "feature_reference"] = "NCBITaxon:10090"
    h5ad_path = f"{assets_path}/{organism.name}_{dataset_id}.h5ad"
    h5ad.write_h5ad(h5ad_path)
    datasets.append(
        Dataset(
            dataset_id=f"{organism.name}_{dataset_id}",
            dataset_title=f"title_{organism.name}",
            collection_id=f"id_{organism.name}",
            collection_name=f"collection_{organism.name}",
            corpora_asset_h5ad_uri="mock",
            dataset_h5ad_path=h5ad_path,
        ),
    )
    return datasets


@pytest.fixture
def datasets_with_larger_raw_layer(assets_path: str) -> List[Dataset]:
    organism = ORGANISMS[0]
    dataset_id = "an_id"
    datasets = []
    h5ad = get_h5ad(organism)
    # Add a raw layer (same as normalized layer)
    h5ad.raw = h5ad
    # Patch the normalized layer so that the last gene is dropped
    h5ad = h5ad[:, 0:3].copy()
    h5ad_path = f"{assets_path}/{organism.name}_{dataset_id}.h5ad"
    h5ad.write_h5ad(h5ad_path)
    datasets.append(
        Dataset(
            dataset_id=f"{organism.name}_{dataset_id}",
            dataset_title=f"title_{organism.name}",
            collection_id=f"id_{organism.name}",
            collection_name=f"collection_{organism.name}",
            corpora_asset_h5ad_uri="mock",
            dataset_h5ad_path=h5ad_path,
        ),
    )
    return datasets
