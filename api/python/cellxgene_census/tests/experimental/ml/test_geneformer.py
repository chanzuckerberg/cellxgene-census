import pytest
import tiledbsoma

import cellxgene_census

try:
    from cellxgene_census.experimental.ml import GeneformerTokenizer
except ImportError:
    # this should only occur when not running `experimental`-marked tests
    pass


def select_census_version_for_geneformer_tests() -> str:
    # GeneformerTokenizer needs the "normalized" X layer which wasn't yet available in
    # "stable" at the time this was written. This should provide a graceful transition
    # once we next advance "stable" (after which it could be eliminated).
    with cellxgene_census.open_soma(census_version="stable") as stable:
        if "normalized" in stable["census_data"]["homo_sapiens"].ms["RNA"].X:
            return "stable"
    return "2023-09-04"


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize(
    "cells_per_chunk",
    [4, 100_000],
)
def test_GeneformerTokenizer(cells_per_chunk: int) -> None:
    # cell soma_joinid: (token sequence length, prefix of token sequence)
    expected_data = {
        1234567: (2048, [15947, 7062, 621, 9291, 9939, 16985, 4113]),
        3703701: (2048, [11180, 11367, 512, 1557, 968, 16411, 16445]),
        8641969: (2011, [3264, 9400, 5485, 2053, 376, 11436, 4533]),
        9876536: (1215, [4285, 6349, 9512, 10856, 520, 7883, 1250]),
        13580237: (2048, [24685, 650, 16997, 15633, 15287, 8121, 13147]),
        14814804: (2048, [16725, 17261, 5368, 16472, 4662, 4737, 11143]),
        16049371: (660, [3707, 10967, 13452, 9538, 13925, 1310, 4093]),
        18518505: (2048, [7711, 9681, 1532, 15633, 14929, 652, 6061]),
        19753072: (2048, [1181, 15982, 4529, 12996, 9061, 3789, 16865]),
        20987639: (1367, [8681, 10047, 9069, 6623, 14968, 16865, 7725]),
        22222206: (2048, [15933, 15623, 8809, 754, 25306, 411, 6872]),
        23456773: (2048, [15633, 11385, 16997, 650, 17184, 3408, 7066]),
        25925907: (1589, [15602, 10824, 3106, 608, 8510, 13232, 24344]),
        28395041: (589, [6556, 19788, 18489, 2124, 10509, 2218, 6567]),
        30864175: (815, [10057, 9500, 2936, 21070, 13659, 5081, 9209]),
        32098742: (2048, [4067, 948, 1324, 5261, 16985, 1511, 10268]),
    }

    with cellxgene_census.open_soma(census_version=select_census_version_for_geneformer_tests()) as census:
        with GeneformerTokenizer(
            census["census_data"]["homo_sapiens"],
            obs_query=tiledbsoma.AxisQuery(coords=(list(expected_data.keys()),)),
            obs_attributes=(
                "soma_joinid",
                "cell_type_ontology_term_id",
                "tissue_ontology_term_id",
            ),
            _cells_per_chunk=cells_per_chunk,  # test with & without pagination
        ) as tokenizer:
            df = tokenizer.build().to_pandas()
            assert len(df) == len(expected_data)
            for row in df.itertuples():
                assert row.length == expected_data[row.soma_joinid][0]
                top_tokens = expected_data[row.soma_joinid][1]
                assert row.input_ids.tolist()[: len(top_tokens)] == top_tokens
                assert row.cell_type_ontology_term_id
                assert row.tissue_ontology_term_id


@pytest.mark.experimental
@pytest.mark.live_corpus
def test_GeneformerTokenizer_docstring_example() -> None:
    with cellxgene_census.open_soma(census_version=select_census_version_for_geneformer_tests()) as census:
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
