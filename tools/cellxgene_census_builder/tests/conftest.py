import pathlib
from functools import partial
from textwrap import dedent
from typing import Any, Literal

import anndata
import attrs
import numpy as np
import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch
from scipy import sparse

from cellxgene_census_builder.build_soma.datasets import Dataset
from cellxgene_census_builder.build_soma.globals import (
    CENSUS_X_LAYERS_PLATFORM_CONFIG,
)
from cellxgene_census_builder.build_state import CensusBuildArgs, CensusBuildConfig
from cellxgene_census_builder.process_init import process_init

MATRIX_FORMAT = Literal["csr", "csc", "dense"]


@attrs.define(frozen=True)
class Organism:
    name: str
    organism_ontology_term_id: str


ORGANISMS = [Organism("homo_sapiens", "NCBITaxon:9606"), Organism("mus_musculus", "NCBITaxon:10090")]
GENE_IDS = [["a", "b", "c", "d"], ["a", "b", "e"], ["a", "b", "c"], ["e", "b", "c", "a"]]
ASSAY_IDS = ["EFO:0009922", "EFO:0008931", "EFO:0009922", "EFO:0008931"]
X_FORMAT: list[MATRIX_FORMAT] = ["csr", "csc", "csr", "dense"]
NUM_DATASET = 4


def get_anndata(
    organism: Organism,
    gene_ids: list[str] | None = None,
    no_zero_counts: bool = False,
    X_format: MATRIX_FORMAT = "csr",
    assay_ontology_term_id: str = "EFO:0009922",
) -> anndata.AnnData:
    gene_ids = gene_ids or GENE_IDS[0]
    n_cells = 4
    n_genes = len(gene_ids)
    rng = np.random.default_rng()
    if no_zero_counts:
        X = rng.integers(1, 6, size=(n_cells, n_genes)).astype(np.float32)
    else:
        X = sparse.random(
            n_cells,
            n_genes,
            density=0.5,
            random_state=rng,
            data_rvs=partial(rng.integers, 1, 6),
            dtype=np.float32,
        ).toarray()

    # Builder code currently assumes (and enforces) that ALL cells (rows) contain at least
    # one non-zero value in their count matrix. Enforce this assumption, as the rng will
    # occasionally generate row that sums to zero.
    X[X.sum(axis=1) == 0, rng.integers(X.shape[1])] = 6.0
    assert np.all(X.sum(axis=1) > 0.0)

    match X_format:
        case "csr":
            X = sparse.csr_matrix(X)
        case "csc":
            X = sparse.csc_matrix(X)
        case "dense":
            pass
        case _:
            raise NotImplementedError("unsupported X format")

    # Create obs
    obs_dataframe = pd.DataFrame(
        data={
            "cell_idx": pd.Series([1, 2, 3, 4]),
            "cell_type_ontology_term_id": "CL:0000192",
            "assay_ontology_term_id": assay_ontology_term_id,
            "disease_ontology_term_id": "PATO:0000461",
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
            "tissue_type": "tissue",
            "observation_joinid": "test",
        },
        index=["1", "2", "3", "4"],
    )
    obs = obs_dataframe

    # Create vars
    feature_id = pd.Series(data=[f"{organism.name}_{g}" for g in gene_ids])
    var_dataframe = pd.DataFrame(
        data={
            "feature_biotype": "gene",
            "feature_is_filtered": False,
            "feature_name": "ERCC-00002 (spike-in control)",
            "feature_type": "synthetic",
            "feature_reference": organism.organism_ontology_term_id,
            "feature_length": 1000,
        },
        index=feature_id,
    )
    var = var_dataframe

    # Create embeddings
    rng.random()
    random_embedding = rng.random(n_cells * n_genes).reshape(n_cells, n_genes)
    obsm = {"X_awesome_embedding": random_embedding}

    # Create uns corpora metadata
    uns: dict[str, Any] = {}
    uns["batch_condition"] = np.array(["a", "b"], dtype="object")

    # Set CxG schema fields (schema 6.0.0: organism fields moved to uns)
    uns["schema_version"] = "6.0.0"
    uns["organism_ontology_term_id"] = organism.organism_ontology_term_id
    uns["organism"] = organism.name

    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm, uns=uns)


@pytest.fixture
def census_build_args(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> CensusBuildArgs:
    # parameterization is optional
    try:
        config = request.param
    except AttributeError:
        config = {}

    # blocklist file is mandatory
    blocklist_path = tmp_path / "blocklist.txt"
    pathlib.Path(blocklist_path).touch()
    config["dataset_id_blocklist_uri"] = blocklist_path.as_posix()

    if config.get("manifest") is True:  # if bool True, replace with an IOstream
        config["manifest"] = request.getfixturevalue("manifest_csv")
    return CensusBuildArgs(working_dir=tmp_path, config=CensusBuildConfig(**config))


@pytest.fixture
def datasets(census_build_args: CensusBuildArgs) -> list[Dataset]:
    census_build_args.h5ads_path.mkdir(parents=True, exist_ok=True)
    assets_path = census_build_args.h5ads_path.as_posix()
    datasets = []
    for organism in ORGANISMS:
        for i in range(NUM_DATASET):
            h5ad = get_anndata(
                organism, GENE_IDS[i], no_zero_counts=False, assay_ontology_term_id=ASSAY_IDS[i], X_format=X_FORMAT[i]
            )
            h5ad_name = f"{organism.name}_{i}.h5ad"
            h5ad.write_h5ad(f"{assets_path}/{h5ad_name}")
            datasets.append(
                Dataset(
                    dataset_id=f"{organism.name}_{i}",
                    dataset_title=f"title_{organism.name}",
                    citation="citation",
                    collection_id=f"id_{organism.name}",
                    collection_name=f"collection_{organism.name}",
                    dataset_asset_h5ad_uri="mock",
                    dataset_h5ad_path=h5ad_name,
                    dataset_version_id=f"{organism.name}_{i}_v0",
                ),
            )

    return datasets


@pytest.fixture
def manifest_csv(tmp_path: pathlib.Path) -> str:
    manifest_content = dedent(f"""\
    dataset_id,dataset_asset_h5ad_uri
    dataset_id_1,{tmp_path}/data/h5ads/dataset_id_1.h5ad
    dataset_id_2,{tmp_path}/data/h5ads/dataset_id_2.h5ad
    """)
    path = f"{tmp_path}/manifest.csv"
    h5ad_path = f"{tmp_path}/data/h5ads/"
    pathlib.Path(h5ad_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmp_path / "data/h5ads/dataset_id_1.h5ad").touch()
    pathlib.Path(tmp_path / "data/h5ads/dataset_id_2.h5ad").touch()
    with open(path, "w+") as f:
        f.writelines(manifest_content.strip())

    return path


@pytest.fixture
def manifest_csv_with_duplicates(tmp_path: pathlib.Path) -> str:
    manifest_content = dedent(f"""\
    dataset_id,dataset_asset_h5ad_uri
    dataset_id_1,{tmp_path}/data/h5ads/dataset_id_1.h5ad
    dataset_id_2,{tmp_path}/data/h5ads/dataset_id_2.h5ad
    dataset_id_2,{tmp_path}/data/h5ads/dataset_id_2.h5ad
    """)
    path = f"{tmp_path}/manifest.csv"
    h5ad_path = f"{tmp_path}/data/h5ads/"
    pathlib.Path(h5ad_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmp_path / "data/h5ads/dataset_id_1.h5ad").touch()
    pathlib.Path(tmp_path / "data/h5ads/dataset_id_2.h5ad").touch()
    with open(path, "w+") as f:
        f.writelines(manifest_content.strip())

    return path


@pytest.fixture
def setup(monkeypatch: MonkeyPatch, census_build_args: CensusBuildArgs) -> None:
    process_init(census_build_args)
    monkeypatch.setitem(CENSUS_X_LAYERS_PLATFORM_CONFIG["raw"]["tiledb"]["create"]["dims"]["soma_dim_0"], "tile", 2)
    monkeypatch.setitem(CENSUS_X_LAYERS_PLATFORM_CONFIG["raw"]["tiledb"]["create"]["dims"]["soma_dim_1"], "tile", 2)


@pytest.fixture
def empty_blocklist(tmp_path: pathlib.Path) -> str:
    blocklist_path = tmp_path / "blocklist.txt"
    pathlib.Path(blocklist_path).touch()
    return blocklist_path.as_posix()
