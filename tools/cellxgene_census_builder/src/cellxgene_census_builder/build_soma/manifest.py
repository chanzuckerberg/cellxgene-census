import csv
import io
import logging
import os.path
from typing import List, Optional, Union

from .datasets import Dataset
from .globals import CXG_SCHEMA_VERSION_IMPORT
from .util import fetch_json

CXG_BASE_URI = "https://api.cellxgene.cziscience.com/"


def parse_manifest_file(manifest_fp: io.TextIOBase) -> List[Dataset]:
    """
    return manifest as list of tuples, (dataset_id, URI/path), read from the text stream
    """
    # skip comments and strip leading/trailing white space
    skip_comments = csv.reader(row for row in manifest_fp if not row.startswith("#"))
    stripped = [[r.strip() for r in row] for row in skip_comments]
    return [Dataset(dataset_id=r[0], dataset_asset_h5ad_uri=r[1]) for r in stripped]


def dedup_datasets(datasets: List[Dataset]) -> List[Dataset]:
    ds = {d.dataset_id: d for d in datasets}
    if len(ds) != len(datasets):
        logging.warning("Dataset manifest contained DUPLICATES, which will be ignored.")
        return list(ds.values())
    return datasets


def load_manifest_from_fp(manifest_fp: io.TextIOBase) -> List[Dataset]:
    logging.info("Loading manifest from file")
    all_datasets = parse_manifest_file(manifest_fp)
    datasets = [
        d
        for d in all_datasets
        if d.dataset_asset_h5ad_uri.endswith(".h5ad") and os.path.exists(d.dataset_asset_h5ad_uri)
    ]
    if len(datasets) != len(all_datasets):
        logging.warning("Manifest contained records which are not H5AD files or which are not accessible - ignoring")
    return datasets


def null_to_empty_str(val: Union[None, str]) -> str:
    if val is None:
        return ""
    return val


def load_manifest_from_CxG() -> List[Dataset]:
    logging.info("Loading manifest from CELLxGENE data portal...")

    # Load all collections and extract dataset_id
    datasets = fetch_json(f"{CXG_BASE_URI}curation/v1/datasets")
    assert isinstance(datasets, list), "Unexpected REST API response, /curation/v1/datasets"

    response = []

    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        schema_version = dataset["schema_version"]

        if schema_version not in CXG_SCHEMA_VERSION_IMPORT:
            logging.warning(f"Dropping dataset {dataset_id} due to unsupported schema version")
            continue

        assets = dataset.get("assets", [])
        assets_h5ad = [a for a in assets if a["filetype"] == "H5AD"]
        if not assets_h5ad:
            logging.error(f"Unable to find H5AD asset for dataset id {dataset_id} - ignoring this dataset")
            continue
        if len(assets_h5ad) > 1:
            logging.error(f"Dataset id {dataset_id} has more than one H5AD asset - ignoring this dataset")
            continue
        asset_h5ad_uri = assets_h5ad[0]["url"]
        asset_h5ad_filesize = assets_h5ad[0]["filesize"]

        d = Dataset(
            dataset_id=dataset_id,
            dataset_asset_h5ad_uri=asset_h5ad_uri,
            dataset_title=null_to_empty_str(dataset["title"]),
            collection_id=dataset["collection_id"],
            collection_name=null_to_empty_str(dataset["collection_name"]),
            collection_doi=null_to_empty_str(dataset["collection_doi"]),
            asset_h5ad_filesize=asset_h5ad_filesize,
            schema_version=schema_version,
        )
        response.append(d)

    logging.info(f"Found {len(datasets)} datasets")

    return response


def load_manifest(manifest_fp: Optional[Union[str, io.TextIOBase]] = None) -> List[Dataset]:
    """
    Load dataset manifest from the file pointer if provided, else bootstrap
    the load rom the CELLxGENE REST API.
    """
    if manifest_fp is not None:
        if isinstance(manifest_fp, str):
            with open(manifest_fp) as f:
                datasets = load_manifest_from_fp(f)
        else:
            datasets = load_manifest_from_fp(manifest_fp)
    else:
        datasets = load_manifest_from_CxG()

    logging.info(f"Loaded {len(datasets)} datasets.")
    datasets = dedup_datasets(datasets)
    return datasets
