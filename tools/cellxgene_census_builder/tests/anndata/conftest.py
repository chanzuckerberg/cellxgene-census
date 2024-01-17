import pathlib
from typing import List

import pytest
from cellxgene_census_builder.build_soma.datasets import Dataset
from cellxgene_census_builder.build_state import CensusBuildArgs

from ..conftest import ORGANISMS, get_anndata


@pytest.fixture
def datasets_with_mixed_feature_reference(census_build_args: CensusBuildArgs) -> List[Dataset]:
    census_build_args.h5ads_path.mkdir(parents=True, exist_ok=True)
    assets_path = census_build_args.h5ads_path.as_posix()

    organism = ORGANISMS[0]
    dataset_id = "an_id"
    datasets = []
    h5ad = get_anndata(organism)
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
            dataset_asset_h5ad_uri="mock",
            dataset_h5ad_path=h5ad_path,
            dataset_version_id=f"{organism.name}_{dataset_id}",
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
    h5ad = get_anndata(organism)
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
            dataset_asset_h5ad_uri="mock",
            dataset_h5ad_path=h5ad_path,
            dataset_version_id=f"{organism.name}_{dataset_id}",
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
    h5ad = get_anndata(organism)
    h5ad.uns["schema_version"] = "2.0.0"
    h5ad_path = f"{assets_path}/{organism.name}_{dataset_id}.h5ad"
    h5ad.write_h5ad(h5ad_path)
    datasets.append(
        Dataset(
            dataset_id=f"{organism.name}_{dataset_id}",
            dataset_title=f"title_{organism.name}",
            collection_id=f"id_{organism.name}",
            collection_name=f"collection_{organism.name}",
            dataset_asset_h5ad_uri="mock",
            dataset_h5ad_path=h5ad_path,
            dataset_version_id=f"{organism.name}_{dataset_id}_v0",
        ),
    )
    return datasets


@pytest.fixture
def h5ad_simple(tmp_path: pathlib.Path) -> str:
    adata = get_anndata(ORGANISMS[0])

    path = "simple.h5ad"
    adata.write_h5ad(tmp_path / path)
    return path


@pytest.fixture
def h5ad_with_organoids_and_cell_culture(tmp_path: pathlib.Path) -> str:
    adata = get_anndata(ORGANISMS[0], no_zero_counts=True)
    adata.obs.at["1", "tissue_ontology_term_id"] = "CL:0000192 (organoid)"
    adata.obs.at["2", "tissue_ontology_term_id"] = "CL:0000192 (cell culture)"

    path = "with_organoids_and_cell_culture.h5ad"
    adata.write_h5ad(tmp_path / path)
    return path


@pytest.fixture
def h5ad_with_organism(tmp_path: pathlib.Path) -> str:
    adata = get_anndata(ORGANISMS[0], no_zero_counts=True)
    adata.obs.at["1", "organism_ontology_term_id"] = ORGANISMS[1].organism_ontology_term_id

    path = "with_organism.h5ad"
    adata.write_h5ad(tmp_path / path)
    return path


@pytest.fixture
def h5ad_with_feature_biotype(tmp_path: pathlib.Path) -> str:
    adata = get_anndata(ORGANISMS[0], no_zero_counts=True)
    adata.var.at["homo_sapiens_c", "feature_biotype"] = "non-gene"

    path = "with_feature_biotype.h5ad"
    adata.write_h5ad(tmp_path / path)
    return path


@pytest.fixture
def h5ad_with_assays(tmp_path: pathlib.Path) -> str:
    adata = get_anndata(ORGANISMS[0], no_zero_counts=True)
    adata.obs.at["1", "assay_ontology_term_id"] = "EFO:1234"
    adata.obs.at["3", "assay_ontology_term_id"] = "EFO:1235"

    path = "with_assays.h5ad"
    adata.write_h5ad(tmp_path / path)
    return path
