import concurrent.futures
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
    return [Dataset(dataset_id=r[0], corpora_asset_h5ad_uri=r[1]) for r in stripped]


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
        if d.corpora_asset_h5ad_uri.endswith(".h5ad") and os.path.exists(d.corpora_asset_h5ad_uri)
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
    collections = fetch_json(f"{CXG_BASE_URI}curation/v1/collections")
    assert isinstance(collections, list), "Unexpected REST API response, /curation/v1/collections"
    datasets = {
        dataset["id"]: {
            "collection_id": collection["id"],
            "collection_name": null_to_empty_str(collection["name"]),
            "collection_doi": null_to_empty_str(collection["doi"]),
            "dataset_title": dataset.get("title", ""),  # title is optional in schema
            "dataset_id": dataset["id"],
        }
        for collection in collections
        for dataset in collection["datasets"]
    }
    logging.info(f"Found {len(datasets)} datasets, in {len(collections)} collections")

    # load per-dataset schema version
    # max_workers currently set to 4 (was 8) and 1 sec delay added per API call due to
    # https://github.com/chanzuckerberg/single-cell-data-portal/issues/3535
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as tp:
        dataset_metadata = tp.map(
            lambda d: fetch_json(
                f"{CXG_BASE_URI}curation/v1/collections/{d['collection_id']}/datasets/{d['dataset_id']}",
                delay_secs=1,
            ),
            datasets.values(),
        )
    for d in dataset_metadata:
        assert (
            isinstance(d, dict) and "id" in d
        ), "Unexpected REST API response, /curation/v1/collections/.../datasets/..."
        datasets[d["id"]].update(
            {
                "schema_version": d["schema_version"],
                "dataset_title": null_to_empty_str(d["title"]),
            }
        )

    # Remove any datasets that don't match our target schema version
    obsolete_dataset_ids = [id for id in datasets if datasets[id]["schema_version"] not in CXG_SCHEMA_VERSION_IMPORT]
    if len(obsolete_dataset_ids) > 0:
        logging.warning(f"Dropping {len(obsolete_dataset_ids)} datasets due to unsupported schema version")
        for id in obsolete_dataset_ids:
            logging.info(f"Dropping dataset_id {id} due to schema version.")
            datasets.pop(id)

    # Grab the asset URI for each dataset
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as tp:
        dataset_assets = tp.map(
            lambda d: (
                d["dataset_id"],
                fetch_json(
                    f"{CXG_BASE_URI}curation/v1/collections/{d['collection_id']}/datasets/{d['dataset_id']}/assets"
                ),
            ),
            datasets.values(),
        )
    no_asset_found = []
    for dataset_id, assets in dataset_assets:
        assert isinstance(
            assets, list
        ), "Unexpected REST API response, /curation/v1/collections/.../datasets/.../assets"
        assets_h5ad = [a for a in assets if a["filetype"] == "H5AD"]
        if len(assets_h5ad) == 0:
            logging.error(f"Unable to find H5AD asset for dataset id {dataset_id} - ignoring this dataset")
            no_asset_found.append(dataset_id)
        else:
            asset = assets_h5ad[0]
            datasets[dataset_id].update(
                {
                    "corpora_asset_h5ad_uri": asset["presigned_url"],
                    "asset_h5ad_filesize": asset["filesize"],
                }
            )

    # drop any datasets where we could not find an asset
    for id in no_asset_found:
        datasets.pop(id, None)

    return [Dataset(**d) for d in datasets.values()]


def load_manifest(manifest_fp: Optional[io.TextIOBase] = None) -> List[Dataset]:
    """
    Load dataset manifest from the file pointer if provided, else bootstrap
    the load rom the CELLxGENE REST API.
    """
    if manifest_fp is not None:
        datasets = load_manifest_from_fp(manifest_fp)
    else:
        datasets = load_manifest_from_CxG()

    logging.info(f"Loaded {len(datasets)} datasets.")
    datasets = dedup_datasets(datasets)
    return datasets
