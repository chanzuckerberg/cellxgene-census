import os
import pathlib
from typing import List

import anndata
import attrs
import numpy as np
import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch
from scipy import sparse

from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.globals import (
    CENSUS_X_LAYERS_PLATFORM_CONFIG,
)
from tools.cell_census_builder.mp import process_initializer


@attrs.define(frozen=True)
class Organism:
    name: str
    organism_ontology_term_id: str


ORGANISMS = [Organism("homo_sapiens", "NCBITaxon:9606"), Organism("mus_musculus", "NCBITaxon:10090")]


def get_h5ad(organism: Organism) -> anndata.AnnData:
    X = np.random.randint(5, size=(4, 4)).astype(np.float32)
    # The builder only supports sparse matrices
    X = sparse.csr_matrix(X)

    # Create obs
    obs_dataframe = pd.DataFrame(
        data={
            "cell_idx": pd.Series([1, 2, 3, 4]),
            "cell_type_ontology_term_id": "CL:0000192",
            "assay_ontology_term_id": "EFO:0008720",
            "disease_ontology_term_id": "PATO:0000461",
            "organism_ontology_term_id": organism.organism_ontology_term_id,
            "sex_ontology_term_id": "unknown",
            "tissue_ontology_term_id": "CL:0000192",
            "is_primary_data": False,
            "self_reported_ethnicity_ontology_term_id": "na",
            "development_stage_ontology_term_id": "MmusDv:0000003",
            "donor_id": "donor_2",
            "suspension_type": "na",
            "assay": "test",
            "cell_type": "test",
            "development_stage": "test",
            "disease": "test",
            "self_reported_ethnicity": "test",
            "sex": "test",
            "tissue": "test",
            "organism": "test",
        },
        index=["1", "2", "3", "4"],
    )
    obs = obs_dataframe

    # Create vars
    feature_name = pd.Series(data=[f"{organism.name}_{g}" for g in ["a", "b", "c", "d"]])
    var_dataframe = pd.DataFrame(
        data={
            "feature_biotype": "gene",
            "feature_is_filtered": False,
            "feature_name": "ERCC-00002 (spike-in control)",
            "feature_reference": organism.organism_ontology_term_id,
        },
        index=feature_name,
    )
    var = var_dataframe

    # Create embeddings
    random_embedding = np.random.rand(4, 4)
    obsm = {"X_awesome_embedding": random_embedding}

    # Create uns corpora metadata
    uns = {}
    uns["batch_condition"] = np.array(["a", "b"], dtype="object")

    # Need to carefully set the corpora schema versions in order for tests to pass.
    uns["schema_version"] = "3.0.0"  # type: ignore

    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm, uns=uns)


@pytest.fixture
def assets_path(tmp_path: pathlib.Path) -> str:
    assets_path = f"{tmp_path}/h5ads"
    os.mkdir(assets_path)
    return assets_path


@pytest.fixture
def soma_path(tmp_path: pathlib.Path) -> str:
    soma_path = f"{tmp_path}/soma"
    os.mkdir(soma_path)
    return soma_path


@pytest.fixture
def datasets(assets_path: str) -> List[Dataset]:
    datasets = []
    for organism in ORGANISMS:
        for dataset_id in range(2):
            h5ad = get_h5ad(organism)
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


@pytest.fixture()
def setup(monkeypatch: MonkeyPatch) -> None:
    process_initializer()
    monkeypatch.setitem(
        CENSUS_X_LAYERS_PLATFORM_CONFIG["raw"]["tiledb"]["create"]["dims"]["soma_dim_0"], "tile", 2  # type: ignore
    )
    monkeypatch.setitem(
        CENSUS_X_LAYERS_PLATFORM_CONFIG["raw"]["tiledb"]["create"]["dims"]["soma_dim_1"], "tile", 2  # type: ignore
    )
