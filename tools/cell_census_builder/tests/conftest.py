import os
import pathlib
from typing import List

import anndata
import numpy as np
import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch
from scipy.sparse import csc_matrix

from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.globals import (
    CENSUS_X_LAYERS_PLATFORM_CONFIG,
)
from tools.cell_census_builder.mp import process_initializer


def h5ad() -> anndata.AnnData:
    X = np.random.randint(5, size=(4, 4))
    # The builder only supports sparse matrices
    X = csc_matrix(X)

    # Create obs
    obs_dataframe = pd.DataFrame(
        data={
            "cell_idx": pd.Series([1, 2, 3, 4]),
            "cell_type_ontology_term_id": "CL:0000192",
            "assay_ontology_term_id": "EFO:0008720",
            "disease_ontology_term_id": "PATO:0000461",
            "organism_ontology_term_id": "NCBITaxon:9606",  # TODO: add one that fails the filter
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
        }
    )
    obs = obs_dataframe

    # Create vars
    feature_name = pd.Series(data=["a", "b", "c", "d"])
    var_dataframe = pd.DataFrame(
        data={
            "feature_biotype": "gene",
            "feature_is_filtered": False,
            "feature_name": "ERCC-00002 (spike-in control)",
            "feature_reference": "NCBITaxon:9606",
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
    first_h5ad = h5ad()
    second_h5ad = h5ad()

    first_h5ad_path = f"{assets_path}/first.h5ad"
    second_h5ad_path = f"{assets_path}/second.h5ad"

    first_h5ad.write_h5ad(first_h5ad_path)
    second_h5ad.write_h5ad(second_h5ad_path)

    return [
        Dataset(dataset_id="first_id", corpora_asset_h5ad_uri="mock", dataset_h5ad_path=first_h5ad_path),
        Dataset(dataset_id="second_id", corpora_asset_h5ad_uri="mock", dataset_h5ad_path=second_h5ad_path),
    ]


@pytest.fixture()
def setup(monkeypatch: MonkeyPatch) -> None:
    process_initializer()
    monkeypatch.setitem(
        CENSUS_X_LAYERS_PLATFORM_CONFIG["raw"]["tiledb"]["create"]["dims"]["soma_dim_0"], "tile", 2  # type: ignore
    )
    monkeypatch.setitem(
        CENSUS_X_LAYERS_PLATFORM_CONFIG["raw"]["tiledb"]["create"]["dims"]["soma_dim_1"], "tile", 2  # type: ignore
    )
