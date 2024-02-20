"""API to facilitate preprocessing of SOMA datasets."""

from ._highly_variable_genes import get_highly_variable_genes, highly_variable_genes
from ._stats import mean_variance

__all__ = [
    "get_highly_variable_genes",
    "highly_variable_genes",
    "mean_variance",
]
