import pickle
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy
import tiledbsoma

from .cell_dataset_builder import CellDatasetBuilder


class GeneformerTokenizer(CellDatasetBuilder):
    """Generate a Hugging Face `Dataset` containing Geneformer token sequences for each
    cell in CELLxGENE Census ExperimentAxisQuery results (human).

    This class requires the Geneformer package to be installed separately with:
    `pip install transformers[torch]<4.50 git+https://huggingface.co/ctheodoris/Geneformer@ebc1e096`

    **DEPRECATION NOTICE:** this is planned for removal from the cellxgene_census API and
    migrated into git:cellxgene-census/tools/models/geneformer.

    Example usage:

    ```
    import cellxgene_census
    import tiledbsoma
    from cellxgene_census.experimental.ml.huggingface import GeneformerTokenizer

    with cellxgene_census.open_soma(census_version="latest") as census:
        with GeneformerTokenizer(
            census["census_data"]["homo_sapiens"],
            # set obs_query to define some subset of Census cells:
            obs_query=tiledbsoma.AxisQuery(value_filter="is_primary_data == True and tissue_general == 'tongue'"),
            obs_column_names=(
                "soma_joinid",
                "cell_type_ontology_term_id",
            ),
        ) as tokenizer:
            dataset = tokenizer.build()
    ```

    Dataset item contents:
    - `input_ids`: Geneformer token sequence for the cell
    - `length`: Length of the token sequence
    - and the specified `obs_column_names` (cell metadata from the experiment obs dataframe)
    """

    obs_column_names: set[str]
    max_input_tokens: int
    special_token: bool

    # Newer versions of Geneformer has a consolidated gene list (gene_mapping_file), meaning the
    # counts for one or more Census genes are to be summed to get the count for one Geneformer
    # gene. model_gene_map is a sparse binary matrix to map a cell vector (or multi-cell matrix) of
    # Census gene counts onto Geneformer gene counts. model_gene_map[i,j] is 1 iff the i'th Census
    # gene count contributes to the j'th Geneformer gene count.
    model_gene_map: scipy.sparse.coo_matrix
    model_gene_tokens: npt.NDArray[np.int64]  # Geneformer token for each column of model_gene_map
    model_gene_medians: npt.NDArray[np.float64]  # float for each column of model_gene_map
    model_cls_token: np.int64 | None = None
    model_eos_token: np.int64 | None = None

    def __init__(
        self,
        experiment: tiledbsoma.Experiment,
        *,
        obs_column_names: Sequence[str] | None = None,
        obs_attributes: Sequence[str] | None = None,
        max_input_tokens: int = 4096,
        special_token: bool = True,
        token_dictionary_file: str = "",
        gene_median_file: str = "",
        gene_mapping_file: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize GeneformerTokenizer.

        Args:
        - `experiment`: Census Experiment to query
        - `obs_query`: obs AxisQuery defining the set of Census cells to process (default all)
        - `obs_column_names`: obs dataframe columns (cell metadata) to propagate into attributes
           of each Dataset item
        - `max_input_tokens`: maximum length of Geneformer input token sequence (default 4096)
        - `special_token`: whether to affix separator tokens to the sequence (default True)
        - `token_dictionary_file`, `gene_median_file`: pickle files supplying the mapping of
          Ensembl human gene IDs onto Geneformer token numbers and median expression values.
          By default, these will be loaded from the Geneformer package.
        - `gene_mapping_file`: optional pickle file with mapping for Census gene IDs to model's
        """
        if obs_attributes:  # old name of obs_column_names
            obs_column_names = obs_attributes

        self.max_input_tokens = max_input_tokens
        self.special_token = special_token
        self.obs_column_names = set(obs_column_names) if obs_column_names else set()
        self._load_geneformer_data(experiment, token_dictionary_file, gene_median_file, gene_mapping_file)
        super().__init__(
            experiment,
            measurement_name="RNA",
            layer_name="raw",
            **kwargs,
        )

    def _load_geneformer_data(
        self,
        experiment: tiledbsoma.Experiment,
        token_dictionary_file: str,
        gene_median_file: str,
        gene_mapping_file: str,
    ) -> None:
        """Load (1) the experiment's genes dataframe and (2) Geneformer's static data
        files for gene tokens and median expression; then, intersect them to compute
        self.model_gene_{ids,tokens,medians}.
        """
        # TODO: this work could be reused for all queries on this experiment

        genes_df = (
            experiment.ms["RNA"]
            .var.read(column_names=["soma_joinid", "feature_id"])
            .concat()
            .to_pandas()
            .set_index("soma_joinid")
        )

        if not (token_dictionary_file and gene_median_file and gene_mapping_file):
            try:
                import geneformer
            except ImportError:
                # pyproject.toml can't express Geneformer git+https dependency
                raise ImportError(
                    "Please install Geneformer with: "
                    "pip install transformers[torch]<4.50 git+https://huggingface.co/ctheodoris/Geneformer@ebc1e096"
                ) from None
            if not token_dictionary_file:
                token_dictionary_file = geneformer.tokenizer.TOKEN_DICTIONARY_FILE
            if not gene_median_file:
                gene_median_file = geneformer.tokenizer.GENE_MEDIAN_FILE
            if not gene_mapping_file:
                gene_mapping_file = geneformer.tokenizer.ENSEMBL_MAPPING_FILE
        with open(token_dictionary_file, "rb") as f:
            gene_token_dict = pickle.load(f)
        with open(gene_median_file, "rb") as f:
            gene_median_dict = pickle.load(f)

        gene_mapping = None
        if gene_mapping_file:
            with open(gene_mapping_file, "rb") as f:
                gene_mapping = pickle.load(f)

        # compute model_gene_{ids,tokens,medians} by joining genes_df with Geneformer's
        # dicts
        map_data = []
        map_i = []
        map_j = []
        model_gene_id_by_ensg: dict[str, int] = {}
        model_gene_count = 0
        model_gene_tokens: list[np.int64] = []
        model_gene_medians: list[np.float64] = []
        for gene_id, row in genes_df.iterrows():
            ensg = row["feature_id"]  # ENSG... gene id, which keys Geneformer's dicts
            if gene_mapping is not None:
                ensg = gene_mapping.get(ensg, ensg)
            if ensg in gene_token_dict:
                if ensg not in model_gene_id_by_ensg:
                    model_gene_id_by_ensg[ensg] = model_gene_count
                    model_gene_count += 1
                    model_gene_tokens.append(gene_token_dict[ensg])
                    model_gene_medians.append(gene_median_dict[ensg])
                map_data.append(1)
                map_i.append(gene_id)
                map_j.append(model_gene_id_by_ensg[ensg])

        self.model_gene_map = scipy.sparse.coo_matrix(
            (map_data, (map_i, map_j)), shape=(genes_df.index.max() + 1, model_gene_count), dtype=bool
        )
        self.model_gene_tokens = np.array(model_gene_tokens, dtype=np.int64)
        self.model_gene_medians = np.array(model_gene_medians, dtype=np.float64)

        assert len(np.unique(self.model_gene_tokens)) == len(self.model_gene_tokens)
        assert np.all(self.model_gene_medians > 0)
        # Geneformer models protein-coding and miRNA genes, so the intersection should
        # be north of 18K.
        assert (
            model_gene_count > 18_000
        ), f"Mismatch between Census gene IDs and Geneformer token dicts (only {model_gene_count} common genes)"

        # Precompute a vector by which we'll multiply each cell's expression vector.
        # The denominator normalizes by Geneformer's median expression values.
        # The numerator 10K factor follows Geneformer's tokenizer; theoretically it doesn't affect
        # affect the rank order, but is probably intended to help with numerical precision.
        self.model_gene_medians_factor = 10_000.0 / self.model_gene_medians

        if self.special_token:
            self.model_cls_token = gene_token_dict["<cls>"]
            self.model_eos_token = gene_token_dict["<eos>"]

    def __enter__(self) -> "GeneformerTokenizer":
        super().__enter__()
        # On context entry, load the necessary cell metadata (obs_df)
        obs_column_names = list(self.obs_column_names)
        if "soma_joinid" not in self.obs_column_names:
            obs_column_names.append("soma_joinid")
        self.obs_df = self.obs(column_names=obs_column_names).concat().to_pandas().set_index("soma_joinid")
        return self

    def cell_item(self, cell_joinid: int, cell_Xrow: scipy.sparse.csr_matrix) -> dict[str, Any]:
        """Given the expression vector for one cell, compute the Dataset item providing
        the Geneformer inputs (token sequence and metadata).
        """
        # Apply model_gene_map to cell_Xrow and normalize with row sum & gene medians.
        # Notice we divide by the total count of the complete row (not only of the projected
        # values); this follows Geneformer's internal tokenizer.
        model_expr = (cell_Xrow * self.model_gene_map).multiply(self.model_gene_medians_factor / cell_Xrow.sum())
        assert isinstance(model_expr, scipy.sparse.coo_matrix), type(model_expr)
        assert model_expr.shape == (1, self.model_gene_map.shape[1])

        # figure the resulting tokens in descending order of model_expr
        # (use sparse model_expr.{col,data} to naturally exclude undetected genes)
        token_order = model_expr.col[np.argsort(-model_expr.data)[: self.max_input_tokens]]
        input_ids = self.model_gene_tokens[token_order]

        if self.special_token:
            # affix special tokens, dropping the last two gene tokens if necessary
            if len(input_ids) == self.max_input_tokens:
                input_ids = input_ids[:-1]
            assert self.model_cls_token is not None
            input_ids = np.insert(input_ids, 0, self.model_cls_token)
            if len(input_ids) == self.max_input_tokens:
                input_ids = input_ids[:-1]
            assert self.model_eos_token is not None
            input_ids = np.append(input_ids, self.model_eos_token)

        ans = {"input_ids": input_ids, "length": len(input_ids)}
        # add the requested obs attributes
        for attr in self.obs_column_names:
            if attr != "soma_joinid":
                ans[attr] = self.obs_df.at[cell_joinid, attr]
            else:
                ans["soma_joinid"] = cell_joinid
        return ans
