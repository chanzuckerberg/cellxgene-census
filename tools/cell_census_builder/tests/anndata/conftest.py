from typing import List

import anndata as ad
import pytest
from cell_census_builder.build_soma.datasets import Dataset
from cell_census_builder.build_state import CensusBuildArgs

from ..conftest import ORGANISMS, get_h5ad


@pytest.fixture
def datasets_with_mixed_feature_reference(census_build_args: CensusBuildArgs) -> List[Dataset]:
    census_build_args.h5ads_path.mkdir(parents=True, exist_ok=True)
    assets_path = census_build_args.h5ads_path.as_posix()

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
def datasets_with_larger_raw_layer(census_build_args: CensusBuildArgs) -> List[Dataset]:
    census_build_args.h5ads_path.mkdir(parents=True, exist_ok=True)
    assets_path = census_build_args.h5ads_path.as_posix()

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


@pytest.fixture
def datasets_with_incorrect_schema_version(census_build_args: CensusBuildArgs) -> List[Dataset]:
    census_build_args.h5ads_path.mkdir(parents=True, exist_ok=True)
    assets_path = census_build_args.h5ads_path.as_posix()

    organism = ORGANISMS[0]
    dataset_id = "an_id"
    datasets = []
    h5ad = get_h5ad(organism)
    h5ad.uns["schema_version"] = "2.0.0"
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
def h5ad_simple() -> ad.AnnData:
    return get_h5ad(ORGANISMS[0])


@pytest.fixture
def h5ad_with_organoids_and_cell_culture() -> ad.AnnData:
    h5ad = get_h5ad(ORGANISMS[0])
    h5ad.obs.at["1", "tissue_ontology_term_id"] = "CL:0000192 (organoid)"
    h5ad.obs.at["2", "tissue_ontology_term_id"] = "CL:0000192 (cell culture)"
    return h5ad


@pytest.fixture
def h5ad_with_organism() -> ad.AnnData:
    h5ad = get_h5ad(ORGANISMS[0])
    h5ad.obs.at["1", "organism_ontology_term_id"] = ORGANISMS[1].organism_ontology_term_id
    return h5ad


@pytest.fixture
def h5ad_with_feature_biotype() -> ad.AnnData:
    h5ad = get_h5ad(ORGANISMS[0])
    h5ad.var.at["homo_sapiens_c", "feature_biotype"] = "non-gene"
    return h5ad


@pytest.fixture
def h5ad_with_assays() -> ad.AnnData:
    h5ad = get_h5ad(ORGANISMS[0])
    h5ad.obs.at["1", "assay_ontology_term_id"] = "EFO:1234"
    h5ad.obs.at["3", "assay_ontology_term_id"] = "EFO:1235"
    return h5ad
