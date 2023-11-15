import datasets
import pytest
import tiledbsoma
from py.path import local as Path

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
def test_GeneformerTokenizer_correctness(tmpdir: Path, N: int) -> None:
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION_FOR_GENEFORMER_TESTS) as census:
        human = census["census_data"]["homo_sapiens"]
        # read obs dataframe to get soma_joinids of all primary cells
        obs_df = (
            human.obs.read(column_names=["soma_joinid"], value_filter="is_primary_data == True").concat().to_pandas()
        )
        # select N at random
        cell_ids = obs_df["soma_joinid"].sample(n=N).tolist()

        # run GeneformerTokenizer on them
        checksum_test = 0
        with GeneformerTokenizer(
            human,
            obs_query=tiledbsoma.AxisQuery(coords=(cell_ids,)),
        ) as tokenizer:
            for it in tokenizer.build():
                lst1 = it["input_ids"]
                checksum_test += hash(tuple(it["input_ids"]))

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
        checksum_true = 0
        for it in datasets.load_from_disk(tmpdir.join("tk.dataset")):
            lst2 = it["input_ids"]
            checksum_true += hash(tuple(it["input_ids"]))

        # verify identical token sequences
        if N == 1:
            assert lst1 == lst2  # for viewing a full diff with pytest -vv
        assert checksum_test == checksum_true


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
