import csv
import io
import logging

import fsspec

from .datasets import Dataset
from .globals import CXG_SCHEMA_VERSION
from .util import fetch_json

logger = logging.getLogger(__name__)

CXG_BASE_URI = "https://api.cellxgene.cziscience.com/"
# The DEV base, occasionally used for testing.
# CXG_BASE_URI = "https://api.cellxgene.dev.single-cell.czi.technology/"


def parse_manifest_file(manifest_fp: io.TextIOBase) -> list[Dataset]:
    """Return manifest as list of tuples, (dataset_id, URI/path), read from the text stream."""
    # skip comments and strip leading/trailing white space
    skip_comments = csv.reader(row for row in manifest_fp if not row.startswith("#"))
    stripped = [[r.strip() for r in row] for row in skip_comments]
    return [Dataset(dataset_id=r[0], dataset_asset_h5ad_uri=r[1]) for r in stripped]


def dedup_datasets(datasets: list[Dataset]) -> list[Dataset]:
    ds = {d.dataset_id: d for d in datasets}
    if len(ds) != len(datasets):
        logger.warning("Dataset manifest contained DUPLICATES, which will be ignored.")
        return list(ds.values())
    return datasets


def load_manifest_from_fp(manifest_fp: io.TextIOBase) -> list[Dataset]:
    logger.info("Loading manifest from file")
    return parse_manifest_file(manifest_fp)


def null_to_empty_str(val: str | None) -> str:
    if val is None:
        return ""
    return val


def load_manifest_from_CxG() -> list[Dataset]:
    logger.info("Loading manifest from CELLxGENE data portal...")

    # Load all collections and extract dataset_id
    datasets = fetch_json(f"{CXG_BASE_URI}curation/v1/datasets?schema_version={CXG_SCHEMA_VERSION}")
    assert isinstance(datasets, list), "Unexpected REST API response, /curation/v1/datasets"

    response = []

    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        schema_version = dataset["schema_version"]

        if schema_version != CXG_SCHEMA_VERSION:
            msg = f"Manifest fetch: dataset {dataset_id} contains unsupported schema version {schema_version}."
            logger.error(msg)
            raise RuntimeError(msg)

        assets = dataset.get("assets", [])
        assets_h5ad = [a for a in assets if a["filetype"] == "H5AD"]
        if not assets_h5ad:
            msg = f"Manifest fetch: unable to find H5AD asset for dataset id {dataset_id} - this should never happen and is likely an upstream bug"
            logger.error(msg)
            raise RuntimeError(msg)
        if len(assets_h5ad) > 1:
            msg = f"Manifest fetch: dataset id {dataset_id} has more than one H5AD asset - this should never happen and is likely an upstream bug"
            logger.error(msg)
            raise RuntimeError(msg)
        asset_h5ad_uri = assets_h5ad[0]["url"]
        asset_h5ad_filesize = assets_h5ad[0]["filesize"]

        d = Dataset(
            dataset_id=dataset_id,
            dataset_asset_h5ad_uri=asset_h5ad_uri,
            dataset_title=null_to_empty_str(dataset.get("title")),
            citation=dataset["citation"],
            collection_id=dataset["collection_id"],
            collection_name=null_to_empty_str(dataset.get("collection_name")),
            collection_doi=null_to_empty_str(dataset.get("collection_doi")),
            collection_doi_label=null_to_empty_str(dataset.get("collection_doi_label")),
            asset_h5ad_filesize=asset_h5ad_filesize,
            schema_version=schema_version,
            dataset_version_id=null_to_empty_str(dataset.get("dataset_version_id")),
            cell_count=dataset["cell_count"],
            mean_genes_per_cell=dataset["mean_genes_per_cell"],
        )
        response.append(d)

    logger.info(f"Found {len(datasets)} datasets")

    return response


def load_blocklist(dataset_id_blocklist_uri: str | None) -> set[str]:
    blocked_dataset_ids: set[str] = set()
    if not dataset_id_blocklist_uri:
        msg = "No dataset blocklist specified - builder is misconfigured"
        logger.error(msg)
        raise ValueError(msg)

    with fsspec.open(dataset_id_blocklist_uri, "rt") as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                # strip blank lines and comments (hash is never in a UUID)
                continue
            blocked_dataset_ids.add(line)

        logger.info(f"Dataset blocklist found, containing {len(blocked_dataset_ids)} ids.")

    return blocked_dataset_ids


def apply_blocklist(datasets: list[Dataset], dataset_id_blocklist_uri: str | None) -> list[Dataset]:
    try:
        blocked_dataset_ids = load_blocklist(dataset_id_blocklist_uri)
        return list(filter(lambda d: d.dataset_id not in blocked_dataset_ids, datasets))

    except FileNotFoundError:
        # Blocklist may not exist, so just skip the filtering in this case
        logger.error("No dataset blocklist found")
        raise


def load_manifest(
    manifest_fp: str | io.TextIOBase | None = None,
    dataset_id_blocklist_uri: str | None = None,
) -> list[Dataset]:
    """Load dataset manifest from the file pointer if provided, else bootstrap
    from the CELLxGENE REST API.  Apply the blocklist if provided.
    """
    if manifest_fp is not None:
        if isinstance(manifest_fp, str):
            with open(manifest_fp) as f:
                datasets = load_manifest_from_fp(f)
        else:
            datasets = load_manifest_from_fp(manifest_fp)
    else:
        datasets = load_manifest_from_CxG()

    datasets = apply_blocklist(datasets, dataset_id_blocklist_uri)
    datasets = dedup_datasets(datasets)
    logger.info(f"After blocklist and dedup, will load {len(datasets)} datasets.")
    return datasets
