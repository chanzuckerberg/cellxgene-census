import pickle
from typing import Any, Dict, Optional, Sequence, Set

import numpy as np
import scipy
import tiledbsoma

from .cell_dataset_builder import CensusCellDatasetBuilder

try:
    import geneformer
except ImportError:
    # pyproject.toml can't express Geneformer git+https dependency
    raise ImportError(
        "Please install Geneformer with: " "pip install git+https://huggingface.co/ctheodoris/Geneformer"
    ) from None

GENEFORMER_MAX_INPUT_TOKENS = 2048


class CensusGeneformerTokenizer(CensusCellDatasetBuilder):
    """
    Generate a Hugging Face `Dataset` containing Geneformer token sequences for each
    cell in CELLxGENE census query results.

    Dataset item contents:
    - `input_ids`: Geneformer token sequence for the cell
    - `length`: Length of the token sequence
    - and specified `cell_attributes` (eg `soma_joinid`, `cell_type_ontology_term_id`)
    """

    cell_attributes: Set[str]
    known_gene_ids: np.ndarray  # set of gene soma_joinids that are known to Geneformer
    known_gene_tokens: np.ndarray  # known_gene_ids pos -> token int64
    known_gene_medians: np.ndarray  # known_gene_ids pos -> float64

    def __init__(
        self,
        query: tiledbsoma.ExperimentAxisQuery,
        *args,
        cell_attributes: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """
        - `query`: defines the Census slice to process
        - `cell_attributes`: cell (obs) column names to include in the Dataset
        """
        self.cell_attributes = set(cell_attributes) if cell_attributes else set()
        cells_column_names = list(self.cell_attributes)
        if "soma_joinid" not in self.cell_attributes:
            cells_column_names.append("soma_joinid")
        super().__init__(
            query,
            *args,
            cells_column_names=cells_column_names,
            genes_column_names=["soma_joinid", "feature_id"],
            **kwargs,
        )
        self.load_geneformer_data()

    def load_geneformer_data(self):
        """
        Load Geneformer's static data files for gene tokens and median expression, then
        use them to compute self.known_gene_{ids,tokens,medians}
        """
        with open(geneformer.tokenizer.TOKEN_DICTIONARY_FILE, "rb") as f:
            gene_token_dict = pickle.load(f)
        with open(geneformer.tokenizer.GENE_MEDIAN_FILE, "rb") as f:
            gene_median_dict = pickle.load(f)

        # compute known_gene_{ids,tokens,medians} by joining genes_df with Geneformer's
        # dicts
        known_gene_ids = []
        known_gene_tokens = []
        known_gene_medians = []
        self.unknown_gene_ids = []
        for gene_id, row in self.genes_df.iterrows():
            ensg = row["feature_id"]  # ENSG... gene id, which keys Geneformer's dicts
            if ensg in gene_token_dict:
                known_gene_ids.append(gene_id)
                known_gene_tokens.append(gene_token_dict[ensg])
                known_gene_medians.append(gene_median_dict[ensg])
            else:
                self.unknown_gene_ids.append(ensg)
        self.known_gene_ids = np.array(known_gene_ids, dtype=np.int64)
        self.known_gene_tokens = np.array(known_gene_tokens, dtype=np.int64)
        self.known_gene_medians = np.array(known_gene_medians, dtype=np.float64)

        assert len(np.unique(self.known_gene_ids)) == len(self.known_gene_ids)
        assert len(np.unique(self.known_gene_tokens)) == len(self.known_gene_tokens)
        assert np.all(self.known_gene_medians > 0)
        # Geneformer models protein-coding and miRNA genes, so the intersection should be somewhere
        # a little north of 20K. (22,092 for Census 2023-07-25)
        assert len(self.known_gene_ids) >= 20_000

    def cell_item(self, cell_id: int, cell_Xrow: scipy.sparse.csr_matrix) -> Dict[str, Any]:
        """
        Given the expression vector for one cell, compute the Dataset item providing
        the Geneformer inputs (token sequence and metadata).
        """
        assert cell_Xrow.shape == (1, self.genes_df.shape[0])
        # project cell_Xrow onto the space of known_gene_ids
        known_counts = cell_Xrow[:, self.known_gene_ids]
        assert isinstance(known_counts, scipy.sparse.csr_matrix), type(known_counts)
        # normalize counts by Geneformer's medians. the 10K factor follows Geneformer's
        # tokenizer to "allocate bits to precision"
        known_expr = known_counts.multiply(10_000).multiply(1.0 / self.known_gene_medians)
        assert isinstance(known_expr, scipy.sparse.coo_matrix), type(known_expr)

        # figure the resulting tokens in descending order of known_expr
        # (use sparse known_expr.{col,data} to naturally exclude undetected genes)
        token_order = known_expr.col[np.argsort(-known_expr.data)]
        input_ids = self.known_gene_tokens[token_order][:GENEFORMER_MAX_INPUT_TOKENS]

        ans = {"input_ids": input_ids, "length": len(input_ids)}
        for attr in self.cell_attributes:
            if attr != "soma_joinid":
                ans[attr] = self.cells_df.at[cell_id, attr]
            else:
                ans["soma_joinid"] = cell_id
        return ans
