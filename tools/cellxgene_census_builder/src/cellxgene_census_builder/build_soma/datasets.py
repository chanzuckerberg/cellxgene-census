import dataclasses
import logging
from typing import TypeVar

import pandas as pd
import pyarrow as pa
import tiledbsoma as soma

from .globals import CENSUS_DATASETS_NAME, CENSUS_DATASETS_TABLE_SPEC

T = TypeVar("T", bound="Dataset")

logger = logging.getLogger(__name__)


@dataclasses.dataclass  # TODO: use attrs
class Dataset:
    """Type used to handle source H5AD datasets read from manifest."""

    # Required
    dataset_id: str  # CELLxGENE dataset_id
    dataset_asset_h5ad_uri: str  # the URI from which we originally read the H5AD asset
    dataset_version_id: str = ""  # CELLxGENE dataset_version_id
    dataset_h5ad_path: str = ""  # set after staging, required by end of process

    # Optional - as reported by REST API
    dataset_title: str = ""  # CELLxGENE dataset title
    citation: str = ""  # CELLxGENE citation
    collection_id: str = ""  # CELLxGENE collection id
    collection_name: str = ""  # CELLxGENE collection name
    collection_doi: str = ""  # CELLxGENE collection doi
    collection_doi_label: str = ""  # CELLxGENE collection doi label
    asset_h5ad_filesize: int = -1
    cell_count: int = -1
    mean_genes_per_cell: float = -1.0

    # Optional, inferred from data if not already known
    schema_version: str = ""  # empty string if version unknown
    dataset_total_cell_count: int = 0  # number of cells in the census by dataset, POST-filter

    # Assigned late in the game, only to datasets we incorporate into the census
    soma_joinid: int = -1

    def __post_init__(self) -> None:
        """Type contracts - downstream code assume these types, so enforce it."""
        for f in dataclasses.fields(self):
            assert isinstance(
                getattr(self, f.name), f.type
            ), f"{f.name} has incorrect type, expected {f.type}, got {type(getattr(self,f.name))}"

    @classmethod
    def to_dataframe(cls: type[T], datasets: list[T]) -> pd.DataFrame:
        if len(datasets) == 0:
            return pd.DataFrame({field.name: pd.Series(dtype=field.type) for field in dataclasses.fields(cls)})

        return pd.DataFrame(datasets)

    @classmethod
    def from_dataframe(cls: type[T], datasets: pd.DataFrame) -> list["Dataset"]:
        return [Dataset(**r) for r in datasets.to_dict("records")]  # type: ignore[misc]


def assign_dataset_soma_joinids(datasets: list[Dataset]) -> None:
    for joinid, dataset in enumerate(datasets):
        dataset.soma_joinid = joinid


def create_dataset_manifest(info_collection: soma.Collection, datasets: list[Dataset]) -> None:
    """Write the Census `census_datasets` dataframe."""
    logger.info("Creating dataset_manifest")
    manifest_df = Dataset.to_dataframe(datasets)
    manifest_df = manifest_df[list(CENSUS_DATASETS_TABLE_SPEC.field_names())]
    if len(manifest_df) == 0:
        return

    schema = CENSUS_DATASETS_TABLE_SPEC.to_arrow_schema(manifest_df)

    # write to a SOMA dataframe
    with info_collection.add_new_dataframe(
        CENSUS_DATASETS_NAME,
        schema=schema,
        index_column_names=["soma_joinid"],
        domain=[(manifest_df["soma_joinid"].min(), manifest_df["soma_joinid"].max())],
    ) as manifest:
        manifest.write(pa.Table.from_pandas(manifest_df, preserve_index=False, schema=schema))
