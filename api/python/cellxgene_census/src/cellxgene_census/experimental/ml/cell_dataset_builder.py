import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional, Sequence

import pandas as pd
import scipy.sparse
import tiledbsoma
from datasets import Dataset

from cellxgene_census.experimental.util import X_sparse_iter


class CensusCellDatasetBuilder(ABC):
    """
    Abstract base class for methods to process CELLxGENE Census query results into a
    Hugging Face Dataset in which each item represents one cell.

    Concrete subclasses should implement `__init__()` and `cell_item()` as discussed
    below.
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

    _cells_per_page: int  # see _paginate_Xrows() method below

    def __init__(
        self,
        query: tiledbsoma.ExperimentAxisQuery,
        layer_name: str = "raw",
        cells_column_names: Optional[Sequence[str]] = None,
        genes_column_names: Optional[Sequence[str]] = None,
        _cells_per_page: int = 100_000,
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
        self.cells_df = self.query.obs(column_names=cells_column_names).concat().to_pandas().set_index("soma_joinid")
        self.genes_df = self.query.var(column_names=genes_column_names).concat().to_pandas().set_index("soma_joinid")
        self._cells_per_page = _cells_per_page

    def build(self, from_generator_kwargs: Optional[Dict[str, Any]] = None) -> Dataset:
        """
        Build the dataset from query results

        - `from_generator_kwargs`: kwargs passed through to `Dataset.from_generator()`
        """

        def gen() -> Generator[Dict[str, Any], None, None]:
            for (page_cell_ids, _), Xpage in X_sparse_iter(
                self.query, self.layer_name, stride=self._cells_per_page, reindex_var=False
            ):
                assert isinstance(Xpage, scipy.sparse.csr_matrix)
                for i, cell_id in enumerate(page_cell_ids):
                    yield self.cell_item(cell_id, Xpage.getrow(i))

        return Dataset.from_generator(_DatasetGeneratorPickleHack(gen), **(from_generator_kwargs or {}))

    @abstractmethod
    def cell_item(self, cell_id: int, Xrow: scipy.sparse.csr_matrix) -> Dict[str, Any]:
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

    def __init__(self, generator: Any, generator_id: Optional[str] = None) -> None:
        self.generator = generator
        self.generator_id = generator_id if generator_id is not None else str(uuid.uuid4())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.generator(*args, **kwargs)

    def __reduce__(self) -> Any:
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")
