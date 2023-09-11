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

        def gen():
            for cell_id, cell_Xrow in self._paginate_Xrows():
                yield self.cell_item(cell_id, cell_Xrow)

        return Dataset.from_generator(_DatasetGeneratorPickleHack(gen), **(from_generator_kwargs or {}))

    @abstractmethod
    def cell_item(self, cell_id: int, cell_Xrow: scipy.sparse.csr_matrix) -> Dict[str, Any]:
        """
        Abstract method to process the X row for one cell into a Dataset item.

        - `cell_id`: The cell `soma_joinid`.
        - `cell_Xrow`: The `X` row for this cell. This csr_matrix has a single row 0,
          equal to the `cell_id` row of the full `X` matrix.
        """
        ...

    def _paginate_Xrows(self):
        """
        Helper for processing the query X matrix row-by-row, with pagination to limit
        peak memory usage for large result sets.

        Ideally ExperimentAxisQuery should handle this for us, see:
            https://github.com/single-cell-data/TileDB-SOMA/issues/1528

        Absent that, our workaround is to paginate the cell soma_joinids found in
        self.cells_df, then execute a sub-query for each page. For each sub-query, we
        load its full X results and then iterate its rows. Repeatedly opening all these
        sub-queries is presumably less efficient than the main query handling the row
        pagination internally.
        """
        # paginate cell ids and iterate pages
        cell_ids = self.cells_df.index.to_numpy()
        cell_ids_pages = np.array_split(cell_ids, np.ceil(len(cell_ids) / self._cells_per_page))
        for cell_ids_page in cell_ids_pages:
            # open sub-query for these cell_ids
            with tiledbsoma.ExperimentAxisQuery(
                self.query.experiment,
                self.query.measurement_name,
                obs_query=tiledbsoma.AxisQuery(coords=(cell_ids_page,)),
                var_query=self.query._matrix_axis_query.var,
            ) as page_query:
                # load full X results for this sub-query, then yield each row
                page_Xcsr = self._load_Xcsr(page_query)
                for cell_id in cell_ids_page:
                    yield (cell_id, page_Xcsr.getrow(cell_id))

    def _load_Xcsr(self, query: tiledbsoma.ExperimentAxisQuery) -> scipy.sparse.csr_matrix:
        """
        Load the full X(layer_name) results of the given query
        """
        Xtbl = query.X(self.layer_name).tables().concat()
        (count, cell_ids, gene_ids) = (np.array(Xtbl.column(col)) for col in ("soma_data", "soma_dim_0", "soma_dim_1"))
        return scipy.sparse.coo_matrix(
            (count, (cell_ids, gene_ids)),
            shape=(
                self.cells_df.index.max() + 1,
                self.genes_df.index.max() + 1,
            ),
        ).tocsr()


class _DatasetGeneratorPickleHack:
    """
    SEE: https://github.com/huggingface/datasets/issues/6194
    """

    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = generator_id if generator_id is not None else str(uuid.uuid4())

    def __call__(self, *args, **kwargs):
        return self.generator(*kwargs, **kwargs)

    def __reduce__(self):
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")
