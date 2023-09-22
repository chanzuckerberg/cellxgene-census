import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional

import scipy.sparse
from datasets import Dataset
from tiledbsoma import Experiment, ExperimentAxisQuery

from cellxgene_census.experimental.util import X_sparse_iter


class CellDatasetBuilder(ExperimentAxisQuery[Experiment], ABC):  # type: ignore
    """
    Abstract base class for methods to process CELLxGENE Census ExperimentAxisQuery
    results into a Hugging Face Dataset in which each item represents one cell.
    Subclasses implement the `cell_item()` method to process each row of an X layer
    into a Dataset item, and may also override `__init__()` and context `__enter__()`
    to perform any necessary preprocessing.

    The base class inherits ExperimentAxisQuery, so typical usage would be:

    ```
    import cellxgene_census
    import tiledbsoma
    from cellxgene_census.experimental.ml import GeneformerTokenizer

    with cellxgene_census.open_soma() as census:
        with SubclassOfCellDatasetBuilder(
            census["census_data"]["homo_sapiens"],
            obs_query=tilebsoma.AxisQuery(...),  # define some subset of Census cells
            ... # other ExperimentAxisQuery parameters e.g. var_query
        ) as builder:
            dataset = builder.build()
    ```
    """

    def __init__(
        self,
        experiment: Experiment,
        measurement_name: str = "RNA",
        layer_name: str = "raw",
        *,
        _cells_per_chunk: int = 100_000,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the CellDatasetBuilder to process the results of a Census
        ExperimentAxisQuery.

        - `experiment`: Census Experiment to be queried.
        - `measurement_name`: Measurement in the experiment, default "RNA".
        - `layer_name`: Name of the X layer to process, default "raw".
        - `verbose`: If True prints progress information.
        - `kwargs`: passed through to `ExperimentAxisQuery()`, especially `obs_query`
           and `var_query`.
        """
        if verbose:
            print("Performing lazy SOMA axis query")
        super().__init__(experiment, measurement_name, **kwargs)
        if verbose:
            print("Lazy query complete")
        self.layer_name = layer_name
        self._cells_per_chunk = _cells_per_chunk
        self.verbose = verbose

    def build(self, from_generator_kwargs: Optional[Dict[str, Any]] = None) -> Dataset:
        """
        Build the dataset from query results

        - `from_generator_kwargs`: kwargs passed through to `Dataset.from_generator()`
        """

        if self.verbose:
            print(f"Initializing Dataset generator")

        def gen() -> Generator[Dict[str, Any], None, None]:
            n_iteration = 1
            total_cells = 0
            if self.verbose:
                print(f"Stepping into Dataset generator. Starting to retrieve data.")
            for (page_cell_joinids, _), Xpage in X_sparse_iter(
                self, self.layer_name, stride=self._cells_per_chunk, reindex_sparse_axis=False
            ):
                assert isinstance(Xpage, scipy.sparse.csr_matrix)
                these_cells = Xpage.shape[0]
                total_cells = total_cells + these_cells
                if self.verbose:
                    print(
                        f"Generating dataset iteration: {n_iteration}. Size: {total_cells} cells. Total: {total_cells}/{self.n_obs}"
                    )
                for i, cell_joinid in enumerate(page_cell_joinids):
                    yield self.cell_item(cell_joinid, Xpage.getrow(i))
                n_iteration += 1

            print(f"Data download complete")

        return Dataset.from_generator(_DatasetGeneratorPickleHack(gen), **(from_generator_kwargs or {}))

    @abstractmethod
    def cell_item(self, cell_joinid: int, Xrow: scipy.sparse.csr_matrix) -> Dict[str, Any]:
        """
        Abstract method to process the X row for one cell into a Dataset item.

        - `cell_joinid`: The cell `soma_joinid`.
        - `Xrow`: The `X` row for this cell. This csr_matrix has a single row 0, equal
          to the `cell_joinid` row of the full `X` layer matrix.
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
