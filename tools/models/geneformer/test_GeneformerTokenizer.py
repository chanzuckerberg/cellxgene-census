# Unit tests for helpers.geneformer_tokenizer; these are executed in the Docker build.
import os
import sys

import cellxgene_census
import datasets
import tiledbsoma
from py.path import local as Path
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))  # to find ./helpers
from geneformer import TranscriptomeTokenizer
from helpers import GeneformerTokenizer

CENSUS_VERSION_FOR_GENEFORMER_TESTS = "2023-12-15"


def test_GeneformerTokenizer_correctness(tmpdir: Path) -> None:
    """Test that GeneformerTokenizer produces the same token sequences as the original
    geneformer.TranscriptomeTokenizer (modulo a small tolerance on Spearman rank correlation).
    """
    # causes deterministic selection of roughly 1,000 cells:
    MODULUS = 32768
    # minimum Spearman rank correlation to consider token sequences effectively identical; this
    # allows for rare, slight differences in token sequences possibly arising from unstable sorting
    # and/or minor numerical precision differences in lowly-expressed genes.
    RHO_THRESHOLD = 0.99
    # notwithstanding RHO_THRESHOLD, we'll check that almost all token sequences are -exactly-
    # identical.
    EXACT_THRESHOLD = 0.98

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION_FOR_GENEFORMER_TESTS) as census:
        human = census["census_data"]["homo_sapiens"]
        # read obs dataframe to get soma_joinids of all primary cells
        obs_df = (
            human.obs.read(column_names=["soma_joinid"], value_filter="is_primary_data == True").concat().to_pandas()
        )
        # select those with soma_joinid == 0 (mod MODULUS)
        cell_ids = [it for it in obs_df["soma_joinid"].tolist() if it % MODULUS == 0]

        # run our GeneformerTokenizer on them
        with GeneformerTokenizer(
            human,
            obs_query=tiledbsoma.AxisQuery(coords=(cell_ids,)),
        ) as tokenizer:
            test_tokens = [it["input_ids"] for it in tokenizer.build()]

        # write h5ad for use with geneformer.TranscriptomeTokenizer
        ad = cellxgene_census.get_anndata(
            census,
            "homo_sapiens",
            X_name="raw",
            obs_coords=cell_ids,
            column_names=tiledbsoma.AxisColumnNames(var=["feature_id"]),
        )
        ad.var.rename(columns={"feature_id": "ensembl_id"}, inplace=True)
        ad.obs["n_counts"] = ad.X.sum(axis=1)
        h5ad_dir = tmpdir.join("h5ad")
        h5ad_dir.mkdir()
        ad.write_h5ad(h5ad_dir.join("tokenizeme.h5ad"))
        # run geneformer.TranscriptomeTokenizer to get "true" tokenizations
        # see: https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/tokenizer.py
        TranscriptomeTokenizer({}).tokenize_data(h5ad_dir, str(tmpdir), "tk", file_format="h5ad")
        true_tokens = [it["input_ids"] for it in datasets.load_from_disk(tmpdir.join("tk.dataset"))]

        # check GeneformerTokenizer sequences against geneformer.TranscriptomeTokenizer's
        assert len(test_tokens) == len(cell_ids)
        assert len(true_tokens) == len(cell_ids)
        identical = 0
        for i, cell_id in enumerate(cell_ids):
            if len(test_tokens[i]) != len(true_tokens[i]):
                assert test_tokens[i] == true_tokens[i]  # to show diff
            rho, _ = spearmanr(test_tokens[i], true_tokens[i])
            if rho < RHO_THRESHOLD:
                # token sequences are too dissimilar; assert exact identity so that pytest -vv will
                # show the complete diff:
                assert (
                    test_tokens[i] == true_tokens[i]
                ), f"Discrepant token sequences for cell soma_joinid={cell_id}; Spearman rho={rho}"
            elif test_tokens[i] == true_tokens[i]:
                identical += 1
        assert identical / len(cell_ids) >= EXACT_THRESHOLD


def test_GeneformerTokenizer_docstring_example() -> None:
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION_FOR_GENEFORMER_TESTS) as census:
        with GeneformerTokenizer(
            census["census_data"]["homo_sapiens"],
            # set obs_query to define some subset of Census cells:
            obs_query=tiledbsoma.AxisQuery(value_filter="is_primary_data == True and tissue_general == 'tongue'"),
            obs_attributes=(
                "soma_joinid",
                "cell_type_ontology_term_id",
            ),
            max_input_tokens=2048,
            special_token=False,
        ) as tokenizer:
            dataset = tokenizer.build()
            assert len(dataset) == 15020
            assert sum(it.length for it in dataset.to_pandas().itertuples()) == 27793772
