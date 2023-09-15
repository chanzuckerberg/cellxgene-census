import uuid
from abc import ABC, abstractmethod
from typing import Any, ContextManager, Dict, Generator, Optional, Sequence, Type

import pandas as pd
import scipy.sparse
import tiledbsoma
from datasets import Dataset

from cellxgene_census.experimental.util import X_sparse_iter


class CellDatasetBuilder(ContextManager["CellDatasetBuilder"], ABC):
    """
    Abstract base class for methods to process CELLxGENE Census query results into a
    Hugging Face Dataset in which each item represents one cell. Subclasses implement
    the `cell_item()` method to process the expression vector for a cell into a Dataset
    item, and may optionally override `__init__()` and `__enter__()` to perform any
    necessary preprocessing.
    """

    cells_df: pd.DataFrame
    """
    Cell metadata, indexed by cell `soma_joinid`. This attribute is available to
    subclasses after the context has been entered.
    """

    genes_df: pd.DataFrame
    """
    Gene metadata, indexed by gene `soma_joinid`. This attribute is available to
    subclasses after the context has been entered.
    """

    def __init__(
        self,
        experiment: tiledbsoma.Experiment,
        *,
        measurement_name: str = "RNA",
        layer_name: str = "raw",
        cells_query: Optional[tiledbsoma.AxisQuery] = None,
        cells_column_names: Optional[Sequence[str]] = None,
        genes_query: Optional[tiledbsoma.AxisQuery] = None,
        genes_column_names: Optional[Sequence[str]] = None,
        _cells_per_page: int = 100_000,
    ):
        """
        Initialize the CellDatasetBuilder to process the results of a Census
        ExperimentAxisQuery.

        - `experiment`: Census Experiment to be queried.
        - `measurement_name`: Measurement in the experiment, default "RNA".
        - `layer_name`: Name of the X layer to process, default "raw".
        - `cells_query`: obs AxisQuery defining the set of cells to process (default all).
        - `cells_column_names`: Columns to include in `self.cells_df` (default all).
        - `genes_query`: var AxisQuery defining the set of genes to process (default all).
        - `genes_column_names`: Columns to include in `self.genes_df` (default all).
        """
        self.experiment = experiment
        self.measurement_name = measurement_name
        self.layer_name = layer_name
        self.cells_query = cells_query
        self.cells_column_names = cells_column_names
        self.genes_query = genes_query
        self.genes_column_names = genes_column_names
        self._cells_per_page = _cells_per_page

    def __enter__(self) -> "CellDatasetBuilder":
        # On context entry, start the query and load cells_df and genes_df
        self.query = self.experiment.axis_query(
            self.measurement_name, obs_query=self.cells_query, var_query=self.genes_query
        )
        self.query.__enter__()

        self.cells_df = (
            self.query.obs(column_names=self.cells_column_names).concat().to_pandas().set_index("soma_joinid")
        )
        self.genes_df = (
            self.query.var(column_names=self.genes_column_names).concat().to_pandas().set_index("soma_joinid")
        )

        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Any) -> None:
        self.query.__exit__(exc_type, exc_val, exc_tb)

    def build(self, from_generator_kwargs: Optional[Dict[str, Any]] = None) -> Dataset:
        """
        Build the dataset from query results

        - `from_generator_kwargs`: kwargs passed through to `Dataset.from_generator()`
        """
        assert isinstance(
            self.query, tiledbsoma.ExperimentAxisQuery
        ), "CellDatasetBuilder.build(): context must be entered"

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
