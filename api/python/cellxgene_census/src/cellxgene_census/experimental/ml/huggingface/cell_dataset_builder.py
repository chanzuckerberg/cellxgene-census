import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

import scipy.sparse
from tiledbsoma import Experiment, ExperimentAxisQuery


class CellDatasetBuilder(ExperimentAxisQuery, ABC):  # type: ignore
    """Abstract base class for methods to process CELLxGENE Census ExperimentAxisQuery
    results into a Hugging Face Dataset in which each item represents one cell.
    Subclasses implement the `cell_item()` method to process each row of an X layer
    into a Dataset item, and may also override `__init__()` and context `__enter__()`
    to perform any necessary preprocessing.

    **DEPRECATION NOTICE:** this is planned for removal from the cellxgene_census API and
    migrated into git:cellxgene-census/tools/models/geneformer.

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
        block_size: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the CellDatasetBuilder to process the results of a Census
        ExperimentAxisQuery.

        - `experiment`: Census Experiment to be queried.
        - `measurement_name`: Measurement in the experiment, default "RNA".
        - `layer_name`: Name of the X layer to process, default "raw".
        - `block_size`: Number of cells to process in-memory at once. If unspecified,
           `tiledbsoma.SparseNDArrayRead.blockwise()` will select a default.
        - `kwargs`: passed through to `ExperimentAxisQuery()`, especially `obs_query`
           and `var_query`.
        """
        super().__init__(experiment, measurement_name, **kwargs)
        self.layer_name = layer_name
        self.block_size = block_size
        warnings.warn(
            "cellxgene_census.experimental.ml.huggingface will be removed from API in an upcoming release and migrated to git:cellxgene-census/tools/models/geneformer",
            FutureWarning,
            stacklevel=2,
        )

    def build(self, from_generator_kwargs: dict[str, Any] | None = None) -> "Dataset":  # type: ignore  # noqa: F821
        """Build the dataset from query results.

        - `from_generator_kwargs`: kwargs passed through to `Dataset.from_generator()`
        """
        # late binding to simplify CI dependencies:
        from datasets import Dataset

        def gen() -> Generator[dict[str, Any], None, None]:
            for Xblock, (block_cell_joinids, _) in (
                self.X(self.layer_name).blockwise(axis=0, reindex_disable_on_axis=[1], size=self.block_size).scipy()
            ):
                assert isinstance(Xblock, scipy.sparse.csr_matrix)
                assert Xblock.shape[0] == len(block_cell_joinids)
                for i, cell_joinid in enumerate(block_cell_joinids):
                    yield self.cell_item(cell_joinid, Xblock.getrow(i))

        return Dataset.from_generator(_DatasetGeneratorPickleHack(gen), **(from_generator_kwargs or {}))

    @abstractmethod
    def cell_item(self, cell_joinid: int, Xrow: scipy.sparse.csr_matrix) -> dict[str, Any]:
        """Abstract method to process the X row for one cell into a Dataset item.

        - `cell_joinid`: The cell `soma_joinid`.
        - `Xrow`: The `X` row for this cell. This csr_matrix has a single row 0, equal
          to the `cell_joinid` row of the full `X` layer matrix.
        """
        ...


class _DatasetGeneratorPickleHack:
    """SEE: https://github.com/huggingface/datasets/issues/6194."""

    def __init__(self, generator: Any, generator_id: str | None = None) -> None:
        self.generator = generator
        self.generator_id = generator_id if generator_id is not None else str(uuid.uuid4())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.generator(*args, **kwargs)

    def __reduce__(self) -> Any:
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")
