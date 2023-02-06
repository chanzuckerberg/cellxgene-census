# General unit tests for cell_census_builder. Intention is to add more fine-grained tests for builder.
from argparse import Namespace
import os
from tempfile import mkstemp, TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import tiledbsoma as soma
import unittest
import anndata

# from tools.cell_census_builder.__main__ import build, make_experiment_builders

import numpy as np

from tools.cell_census_builder.__main__ import build, make_experiment_builders
from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.globals import CENSUS_DATA_NAME
from tools.cell_census_builder.mp import process_initializer
from tools.cell_census_builder.util import uricat


class TestBuilder(unittest.TestCase):

    def generate_h5ad(self):
        X = np.random.randint(5, size=(4, 4))

        # Create obs
        obs_dataframe = pd.DataFrame(
            data={
                "cell_idx": pd.Series([1,2,3,4]),
                "cell_type_ontology_term_id": "CL:0000192",
                "assay_ontology_term_id": "EFO:0008720",
                "disease_ontology_term_id": "PATO:0000461",
                "organism_ontology_term_id": "NCBITaxon:9606", # TODO: add one that fails the filter
                "sex_ontology_term_id": "unknown",
                "tissue_ontology_term_id": "CL:0000192",
                "is_primary_data":  False,
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

        print(obs)

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
        uns["schema_version"] = "3.0.0"

        return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm, uns=uns)

    def setUp(self) -> None:

        self.td = TemporaryDirectory()

        self.assets_path = f"{self.td.name}/h5ads"
        os.mkdir(self.assets_path)
        self.soma_path = f"{self.td.name}/soma"
        os.mkdir(self.soma_path)

        first_h5ad = self.generate_h5ad()
        second_h5ad = self.generate_h5ad()

        first_h5ad_path = f"{self.assets_path}/first.h5ad"
        second_h5ad_path = f"{self.assets_path}/second.h5ad"

        first_h5ad.write_h5ad(first_h5ad_path)
        second_h5ad.write_h5ad(second_h5ad_path)

        self.datasets = [
            Dataset(dataset_id = "first_id", corpora_asset_h5ad_uri = "mock", dataset_h5ad_path=first_h5ad_path),
            Dataset(dataset_id = "second_id", corpora_asset_h5ad_uri = "mock", dataset_h5ad_path=second_h5ad_path)
        ]

        process_initializer()

        return super().setUp()


    def tearDown(self) -> None:
        self.td.cleanup()
        return super().tearDown()

    def test_something(self):
        experiment_builders = make_experiment_builders(uricat(self.soma_path, CENSUS_DATA_NAME), [])
        with patch("tools.cell_census_builder.__main__.build_step1_get_source_assets") as m:
            m.return_value = self.datasets
            from types import SimpleNamespace
            args = SimpleNamespace(multi_process=False, consolidate=False, build_tag="test_tag")
            builder = build(args, self.soma_path, self.assets_path, experiment_builders)





if __name__ == '__main__':
    unittest.main()