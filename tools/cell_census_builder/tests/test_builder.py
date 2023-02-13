import os
from tempfile import TemporaryDirectory
from typing import List
from unittest.mock import patch

import anndata
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import tiledb
import tiledbsoma as soma
from scipy.sparse import csc_matrix

from tools.cell_census_builder.__main__ import build, make_experiment_builders
from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.globals import (
    CENSUS_DATA_NAME,
    CENSUS_INFO_NAME,
    CENSUS_X_LAYERS_PLATFORM_CONFIG,
)
from tools.cell_census_builder.mp import process_initializer
from tools.cell_census_builder.util import uricat


class TestBuilder:
    soma_path: str
    assets_path: str
    td: TemporaryDirectory  # type: ignore

    @classmethod
    def setup_class(cls) -> None:
        """
        Setup method that:
        1. Initializes a temporary file system with the correct paths
        2. Patches some configuration to work with fixtures
        3. Calls `process_initializer()` to setup the environment
        """
        cls.td = TemporaryDirectory()

        cls.assets_path = f"{cls.td.name}/h5ads"
        os.mkdir(cls.assets_path)
        cls.soma_path = f"{cls.td.name}/soma"
        os.mkdir(cls.soma_path)

        process_initializer()

        # The tile extent needs to be smaller than the default (2048) to work with the fixture
        CENSUS_X_LAYERS_PLATFORM_CONFIG["raw"]["tiledb"]["create"]["dims"]["soma_dim_0"]["tile"] = 2  # type: ignore
        CENSUS_X_LAYERS_PLATFORM_CONFIG["raw"]["tiledb"]["create"]["dims"]["soma_dim_1"]["tile"] = 2  # type: ignore

    @classmethod
    def teardown_class(cls) -> None:
        cls.td.cleanup()

    @pytest.fixture
    def h5ad(self) -> anndata.AnnData:
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
    def datasets(self, h5ad: anndata.AnnData) -> List[Dataset]:
        first_h5ad = h5ad
        second_h5ad = h5ad

        first_h5ad_path = f"{self.assets_path}/first.h5ad"
        second_h5ad_path = f"{self.assets_path}/second.h5ad"

        first_h5ad.write_h5ad(first_h5ad_path)
        second_h5ad.write_h5ad(second_h5ad_path)

        return [
            Dataset(dataset_id="first_id", corpora_asset_h5ad_uri="mock", dataset_h5ad_path=first_h5ad_path),
            Dataset(dataset_id="second_id", corpora_asset_h5ad_uri="mock", dataset_h5ad_path=second_h5ad_path),
        ]

    def test_base_builder_creation(self, datasets: List[Dataset]) -> None:
        """
        Runs the builder, queries the census and performs a set of base assertions.
        """

        with patch("tools.cell_census_builder.__main__.prepare_file_system"), patch(
            "tools.cell_census_builder.__main__.build_step1_get_source_assets"
        ) as get_assets:
            get_assets.return_value = datasets

            experiment_builders = make_experiment_builders(uricat(self.soma_path, CENSUS_DATA_NAME), [])  # type: ignore
            from types import SimpleNamespace

            args = SimpleNamespace(multi_process=False, consolidate=False, build_tag="test_tag")
            return_value = build(args, self.soma_path, self.assets_path, experiment_builders)  # type: ignore

            # return_value = 0 means that the validation passed
            assert return_value == 0

            # Query the census and do assertions
            census = soma.Collection(
                uri=self.soma_path,
                context=soma.options.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({"vfs.s3.region": "us-west-2"})),
            )

            # There are 8 cells in total (4 from the first and 4 from the second datasets). They all belong to homo_sapiens
            human_obs = census[CENSUS_DATA_NAME]["homo_sapiens"]["obs"].read().concat().to_pandas()
            assert human_obs.shape[0] == 8

            # mus_musculus should have 0 cells
            mouse_obs = census[CENSUS_DATA_NAME]["mus_musculus"]["obs"].read().concat().to_pandas()
            assert mouse_obs.shape[0] == 0

            # There are only 4 unique genes
            var = census[CENSUS_DATA_NAME]["homo_sapiens"]["ms"]["RNA"]["var"].read().concat().to_pandas()
            assert var.shape[0] == 4

            # There should be 2 datasets
            returned_datasets = census[CENSUS_INFO_NAME]["datasets"].read().concat().to_pandas()
            assert returned_datasets.shape[0] == 2
            assert list(returned_datasets["dataset_id"]) == ["first_id", "second_id"]

    def test_unicode_support(self) -> None:
        """
        Regression test that unicode is supported correctly in tiledbsoma.
        This test is not strictly necessary, but it validates the requirements that Cell Census
        support unicode in DataFrame columns.
        """
        with TemporaryDirectory() as d:
            pd_df = pd.DataFrame(data={"value": ["Ünicode", "S̈upport"]}, columns=["value"])
            pd_df["soma_joinid"] = pd_df.index
            s_df = soma.DataFrame(uri=os.path.join(d, "unicode_support")).create(
                pa.Schema.from_pandas(pd_df, preserve_index=False), index_column_names=["soma_joinid"]
            )
            s_df.write(pa.Table.from_pandas(pd_df, preserve_index=False))

            pd_df_in = soma.DataFrame(uri=os.path.join(d, "unicode_support")).read().concat().to_pandas()

            assert pd_df_in["value"].to_list() == ["Ünicode", "S̈upport"]
