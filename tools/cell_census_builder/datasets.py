import dataclasses
import logging
from typing import List, Type, TypeVar

import pandas as pd
import pyarrow as pa
import tiledbsoma as soma

from .globals import CENSUS_DATASETS_COLUMNS, CENSUS_DATASETS_NAME

T = TypeVar("T", bound="Dataset")


@dataclasses.dataclass  # TODO: use attrs
class Dataset:
    """
    Type used to handle source H5AD datasets read from manifest
    """

    # Required
    dataset_id: str  # CELLxGENE dataset_id
    corpora_asset_h5ad_uri: str  # the URI from which we originally read the H5AD asset
    dataset_h5ad_path: str = ""  # set after staging, required by end of process

    # Optional
    dataset_title: str = ""  # CELLxGENE dataset title
    collection_id: str = ""  # CELlxGENE collection id
    collection_name: str = ""  # CELlxGENE collection name
    collection_doi: str = ""  # CELLxGENE collection doi
    asset_h5ad_filesize: int = -1

    # Optional, inferred from data if not already known
    schema_version: str = ""  # empty string if version unknown
    dataset_total_cell_count: int = 0  # number of cells in the census by dataset

    # Assigned late in the game, only to datasets we incorporate into the census
    soma_joinid: int = -1

    def __post_init__(self) -> None:
        """
        Type contracts - downstream code assume these types, so enforce it.
        """
        for f in dataclasses.fields(self):
            assert isinstance(
                getattr(self, f.name), f.type
            ), f"{f.name} has incorrect type, expected {f.type}, got {type(getattr(self,f.name))}"

    @classmethod
    def to_dataframe(cls: Type[T], datasets: List[T]) -> pd.DataFrame:
        if len(datasets) == 0:
            return pd.DataFrame({field.name: pd.Series(dtype=field.type) for field in dataclasses.fields(cls)})

        return pd.DataFrame(datasets)

    @classmethod
    def from_dataframe(cls: Type[T], datasets: pd.DataFrame) -> List["Dataset"]:
        return [Dataset(**r) for r in datasets.to_dict("records")]


def assign_dataset_soma_joinids(datasets: List[Dataset]) -> None:
    for joinid, dataset in enumerate(datasets):
        dataset.soma_joinid = joinid


def create_dataset_manifest(info_collection: soma.Collection, datasets: List[Dataset]) -> None:
    """
    Write the Cell Census `census_datasets` dataframe
    """
    logging.info("Creating dataset_manifest")
    manifest_df = Dataset.to_dataframe(datasets)
    manifest_df = manifest_df[CENSUS_DATASETS_COLUMNS + ["soma_joinid"]]

    # write to a SOMA dataframe
    with info_collection.add_new_dataframe(
        CENSUS_DATASETS_NAME,
        schema=pa.Schema.from_pandas(manifest_df, preserve_index=False),
        index_column_names=["soma_joinid"],
    ) as manifest:
        manifest.write(pa.Table.from_pandas(manifest_df, preserve_index=False))
