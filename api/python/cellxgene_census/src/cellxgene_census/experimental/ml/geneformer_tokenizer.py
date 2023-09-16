import pickle
from typing import Any, Dict, Optional, Sequence, Set

import numpy as np
import numpy.typing as npt
import scipy
import tiledbsoma

from .cell_dataset_builder import CellDatasetBuilder

GENEFORMER_MAX_INPUT_TOKENS = 2048


class GeneformerTokenizer(CellDatasetBuilder):
    """
    Generate a Hugging Face `Dataset` containing Geneformer token sequences for each
    cell in CELLxGENE Census ExperimentAxisQuery results (human).

    This class requires the Geneformer package to be installed separately with:
    `pip install git+https://huggingface.co/ctheodoris/Geneformer@39ab62e`

    Example usage:

    ```
    import cellxgene_census
    import tiledbsoma
    from cellxgene_census.experimental.ml import GeneformerTokenizer

    with cellxgene_census.open_soma() as census:
        with GeneformerTokenizer(
            census["census_data"]["homo_sapiens"],
            obs_query=tiledbsoma.AxisQuery(...),  # define some subset of Census cells
            obs_attributes=(
                "soma_joinid",
                "cell_type_ontology_term_id",
            ),
        ) as tokenizer:
            dataset = tokenizer.build()
    ```

    Dataset item contents:
    - `input_ids`: Geneformer token sequence for the cell
    - `length`: Length of the token sequence
    - and the specified `obs_attributes`
    """

    obs_attributes: Set[str]
    # set of gene soma_joinids corresponding to genes modeled by Geneformer:
    model_gene_ids: npt.NDArray[np.int64]
    model_gene_tokens: npt.NDArray[np.int64]  # token for each model_gene_id
    model_gene_medians: npt.NDArray[np.float64]  # float for each model_gene_id

    def __init__(
        self,
        experiment: tiledbsoma.Experiment,
        *,
        obs_attributes: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        - `experiment`: Census Experiment to query
        - `obs_query`: obs AxisQuery defining the set of Census cells to process (default all)
        - `obs_attributes`: names of obs (cell) attributes to propagate into each Dataset item
        """
        self.load_geneformer_data(experiment)
        self.obs_attributes = set(obs_attributes) if obs_attributes else set()
        super().__init__(
            experiment,
            measurement_name="RNA",
            layer_name="normalized",
            # set up the query to fetch the relevant genes only
            var_query=tiledbsoma.AxisQuery(coords=(self.model_gene_ids,)),
            **kwargs,
        )

    def load_geneformer_data(self, experiment: tiledbsoma.Experiment) -> None:
        """
        Load (1) the experiment's genes dataframe and (2) Geneformer's static data
        files for gene tokens and median expression; then, intersect them to compute
        self.model_gene_{ids,tokens,medians}

        TODO: this could be reused by multiple GeneformerTokenizer instances
        """
        genes_df = experiment.ms["RNA"].var.read(column_names=["soma_joinid", "feature_id"]).concat().to_pandas()

        try:
            import geneformer
        except ImportError:
            # pyproject.toml can't express Geneformer git+https dependency
            raise ImportError(
                "Please install Geneformer with: "
                "pip install git+https://huggingface.co/ctheodoris/Geneformer@39ab62e"
            ) from None
        with open(geneformer.tokenizer.TOKEN_DICTIONARY_FILE, "rb") as f:
            gene_token_dict = pickle.load(f)
        with open(geneformer.tokenizer.GENE_MEDIAN_FILE, "rb") as f:
            gene_median_dict = pickle.load(f)

        # compute model_gene_{ids,tokens,medians} by joining genes_df with Geneformer's
        # dicts
        model_gene_ids = []
        model_gene_tokens = []
        model_gene_medians = []
        for gene_id, row in genes_df.iterrows():
            ensg = row["feature_id"]  # ENSG... gene id, which keys Geneformer's dicts
            if ensg in gene_token_dict:
                model_gene_ids.append(gene_id)
                model_gene_tokens.append(gene_token_dict[ensg])
                model_gene_medians.append(gene_median_dict[ensg])
        self.model_gene_ids = np.array(model_gene_ids, dtype=np.int64)
        self.model_gene_tokens = np.array(model_gene_tokens, dtype=np.int64)
        self.model_gene_medians = np.array(model_gene_medians, dtype=np.float64)

        assert len(np.unique(self.model_gene_ids)) == len(self.model_gene_ids)
        assert len(np.unique(self.model_gene_tokens)) == len(self.model_gene_tokens)
        assert np.all(self.model_gene_medians > 0)
        # Geneformer models protein-coding and miRNA genes, so the intersection should
        # be somewhere a little north of 20K.
        assert len(self.model_gene_ids) >= 20_000

    def __enter__(self) -> "GeneformerTokenizer":
        super().__enter__()
        # On context entry, load the necessary cell metadata (obs_df)
        obs_column_names = list(self.obs_attributes)
        if "soma_joinid" not in self.obs_attributes:
            obs_column_names.append("soma_joinid")
        self.obs_df = self.obs(column_names=obs_column_names).concat().to_pandas().set_index("soma_joinid")
        return self

    def cell_item(self, cell_joinid: int, cell_Xrow: scipy.sparse.csr_matrix) -> Dict[str, Any]:
        """
        Given the expression vector for one cell, compute the Dataset item providing
        the Geneformer inputs (token sequence and metadata).
        """
        # project cell_Xrow onto model_gene_ids
        model_counts = cell_Xrow[:, self.model_gene_ids]
        assert isinstance(model_counts, scipy.sparse.csr_matrix), type(model_counts)
        # normalize counts by Geneformer's medians. the 10K factor follows Geneformer's
        # tokenizer to "allocate bits to precision"
        model_expr = model_counts.multiply(10_000).multiply(1.0 / self.model_gene_medians)
        assert isinstance(model_expr, scipy.sparse.coo_matrix), type(model_expr)

        # figure the resulting tokens in descending order of model_expr
        # (use sparse model_expr.{col,data} to naturally exclude undetected genes)
        token_order = model_expr.col[np.argsort(-model_expr.data)]
        input_ids = self.model_gene_tokens[token_order][:GENEFORMER_MAX_INPUT_TOKENS]

        ans = {"input_ids": input_ids, "length": len(input_ids)}
        for attr in self.obs_attributes:
            if attr != "soma_joinid":
                ans[attr] = self.obs_df.at[cell_joinid, attr]
            else:
                ans["soma_joinid"] = cell_joinid
        return ans
