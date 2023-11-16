import datasets
import pytest
import tiledbsoma
from py.path import local as Path
from scipy.stats import spearmanr

import cellxgene_census

try:
    from geneformer import TranscriptomeTokenizer

    from cellxgene_census.experimental.ml.huggingface import GeneformerTokenizer
except ImportError:
    # this should only occur when not running `experimental`-marked tests
    pass


CENSUS_VERSION_FOR_GENEFORMER_TESTS = "2023-10-23"


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize("N", [100])
@pytest.mark.parametrize("rho_threshold", [0.995])
def test_GeneformerTokenizer_correctness(tmpdir: Path, N: int, rho_threshold: float) -> None:
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION_FOR_GENEFORMER_TESTS) as census:
        human = census["census_data"]["homo_sapiens"]
        # read obs dataframe to get soma_joinids of all primary cells
        obs_df = (
            human.obs.read(column_names=["soma_joinid"], value_filter="is_primary_data == True").concat().to_pandas()
        )
        # select N at random
        cell_ids = obs_df["soma_joinid"].sample(n=N).tolist()

        # run GeneformerTokenizer on them
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
        # run geneformer.TranscriptomeTokenizer
        # see: https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/tokenizer.py
        TranscriptomeTokenizer({}).tokenize_data(h5ad_dir, tmpdir, "tk", file_format="h5ad")
        true_tokens = [it["input_ids"] for it in datasets.load_from_disk(tmpdir.join("tk.dataset"))]

        # verify identical token sequences
        assert len(test_tokens) == len(cell_ids)
        assert len(true_tokens) == len(cell_ids)

        identical = 0
        for i, cell_id in enumerate(cell_ids):
            assert len(test_tokens[i]) == len(true_tokens[i])
            # check rank correlation between test_tokens[i] and true_tokens[i]; this tolerates
            # rare, slight differences in the token sequences which may arise from numerical
            # precision issues in ranking lowly-expressed genes
            rho, _ = spearmanr(test_tokens[i], true_tokens[i])
            if rho < rho_threshold:
                # token sequences are too dissimilar; assert exact identity so that pytest -vv will
                # show the complete diff:
                assert (
                    test_tokens[i] == true_tokens[i]
                ), f"Discrepant token sequences for cell soma_joinid={cell_id}; Spearman rho={rho}"
            elif test_tokens[i] == true_tokens[i]:
                identical += 1

        # notwithstanding the rho_threshold tolerance, verify that almost all sequences are indeed
        # exactly identical
        assert identical / len(cell_ids) >= 0.95


@pytest.mark.experimental
@pytest.mark.live_corpus
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
        ) as tokenizer:
            dataset = tokenizer.build()
            assert len(dataset) == 15020
            assert sum(it.length for it in dataset.to_pandas().itertuples()) == 27798388
