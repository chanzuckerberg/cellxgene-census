import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse
import tiledbsoma
from datasets import Dataset


class CensusCellDatasetBuilder(ABC):
    """
    Abstract base class for methods to process CELLxGENE Census query results into a
    Hugging Face Dataset in which each item represents one cell.
    """

    query: tiledbsoma.ExperimentAxisQuery
    """
    The Census query whose results will be processed. This attribute is available to
    subclasses following base class initialization.
    """

    cells_df: pd.DataFrame
    """
    Cell metadata, indexed by cell `soma_joinid`. This attribute is available to
    subclasses following base class initialization.
    """

    genes_df: pd.DataFrame
    """
    Gene metadata, indexed by gene `soma_joinid`. This attribute is available to
    subclasses following base class initialization.
    """

    def __init__(
        self,
        query: tiledbsoma.ExperimentAxisQuery,
        layer_name: str = "raw",
        cells_column_names: Optional[Sequence[str]] = None,
        genes_column_names: Optional[Sequence[str]] = None,
    ):
        """
        Initialize the CensusCellDatasetBuilder to process the results of the given
        Census query.

        After invoking `super().__init__(...)`, subclass initializers should perform
        any preprocessing of `self.cells_df` and `self.genes_df` that may be needed to
        then process the individual cells efficiently.

        - `layer_name`: Name of the X layer to process, default "raw".
        - `cells_column_names`: Columns to include in `self.cells_df`; all columns by
          default.
        - `genes_column_names`: Columns to include in `self.genes_df`; all columns by
          default.
        """
        self.query = query
        self.layer_name = layer_name
        self.cells_df = (
            self.query.obs(column_names=cells_column_names)
            .concat()
            .to_pandas()
            .set_index("soma_joinid")
        )
        self.genes_df = (
            self.query.var(column_names=genes_column_names)
            .concat()
            .to_pandas()
            .set_index("soma_joinid")
        )

    def build(self) -> Dataset:
        """Build the dataset from query results"""

        # We're using a generator approach to anticipate processing a large X slice
        # without loading it all into memory at once. While we do in fact load it all
        # right now, that's an implementation detail of this ABC which can be improved
        # without breaking subclasses.
        def gen():
            X_tbl = self.query.X(self.layer_name).tables().concat()
            (count, cell_ids, gene_ids) = (
                np.array(X_tbl.column(col))
                for col in ("soma_data", "soma_dim_0", "soma_dim_1")
            )
            X_csr = scipy.sparse.coo_matrix(
                (count, (cell_ids, gene_ids)),
                shape=(
                    self.cells_df.index.max() + 1,
                    self.genes_df.index.max() + 1,
                ),
            ).tocsr()

            for cell_id in np.unique(cell_ids):
                yield self.cell_item(cell_id, X_csr.getrow(cell_id))

        return Dataset.from_generator(_DatasetGeneratorPickleHack(gen))

    @abstractmethod
    def cell_item(
        self, cell_id: int, cell_Xrow: scipy.sparse.csr_matrix
    ) -> Dict[str, Any]:
        """
        Abstract method to process the X row for one cell into a Dataset item.

        - `cell_id`: The cell `soma_joinid`.
        - `cell_Xrow`: The `X` row for this cell. This csr_matrix has a single row 0,
          equal to the `cell_id` row of the full `X` matrix.
        """
        ...


class _DatasetGeneratorPickleHack:
    """
    SEE: https://github.com/huggingface/datasets/issues/6194
    """

    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *args, **kwargs):
        return self.generator(*kwargs, **kwargs)

    def __reduce__(self):
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")
