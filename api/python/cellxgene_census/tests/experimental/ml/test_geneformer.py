import pytest
import tiledbsoma

import cellxgene_census

try:
    from cellxgene_census.experimental.ml import CensusGeneformerTokenizer
except ImportError:
    # this should only occur when not running `experimental`-marked tests
    pass


@pytest.mark.experimental
@pytest.mark.parametrize(
    "cells_per_page",
    [4, 100_000],
)
def test_CensusGeneformerTokenizer(cells_per_page):
    # cell soma_joinid: (token sequence length, prefix of token sequence)
    expected_data = {
        4938268: (1611, [4913, 8981, 3414, 10509, 12175, 3287, 3190]),
        6172835: (2048, [23812, 12734, 12247, 11828, 11817, 11856]),
        7407402: (337, [3705, 10199, 9717, 2855, 4968, 12379, 23]),
        8641969: (1492, [14594, 8225, 6252, 14127, 17272, 3812, 1868]),
        12345670: (1666, [4612, 6064, 4135, 4662, 3584, 2649, 1435]),
        14814804: (1162, [6645, 15022, 12223, 5841, 9057, 15005, 1480]),
        16049371: (590, [2690, 8091, 9280, 22723, 12266, 15678, 6764]),
        18518505: (1738, [7066, 6712, 10835, 11385, 1142, 6531, 16997]),
        19753072: (2048, [650, 14751, 15633, 10144, 8614, 4114, 9362]),
        22222206: (1244, [1106, 4067, 20475, 4113, 15947, 5213, 9939]),
        28395041: (482, [2218, 6050, 5213, 14075, 11828, 9500, 4488]),
        29629608: (2048, [20041, 5648, 18941, 10292, 14738, 4128, 12698]),
        33333309: (1955, [608, 15243, 8149, 2657, 7055, 8446, 8251]),
        37037010: (611, [22562, 10057, 20334, 1147, 2708, 8743, 16203]),
        38271577: (2048, [2526, 16979, 11347, 9276, 4703, 11450, 2218]),
        39506144: (1427, [4809, 5538, 19788, 14127, 18489, 9945, 20031]),
    }

    with cellxgene_census.open_soma(census_version="2023-07-25") as census:
        human = census["census_data"]["homo_sapiens"]
        with human.axis_query(
            measurement_name="RNA",
            obs_query=tiledbsoma.AxisQuery(coords=(list(expected_data.keys()),)),
        ) as query:
            tokenizer = CensusGeneformerTokenizer(
                query,
                cell_attributes=(
                    "soma_joinid",
                    "cell_type_ontology_term_id",
                    "tissue_ontology_term_id",
                ),
            )
            tokenizer._cells_per_page = cells_per_page  # test with & without pagination
            df = tokenizer.build().to_pandas()
            assert len(df) == len(expected_data)
            for row in df.itertuples():
                assert row.length == expected_data[row.soma_joinid][0]
                top_tokens = expected_data[row.soma_joinid][1]
                assert row.input_ids.tolist()[: len(top_tokens)] == top_tokens
