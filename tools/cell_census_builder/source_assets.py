import argparse
import logging
import os
import urllib.parse
from typing import List, Tuple, cast

import aiohttp
import fsspec

from .datasets import Dataset
from .mp import cpu_count, create_process_pool_executor


def stage_source_assets(datasets: List[Dataset], args: argparse.Namespace, assets_dir: str) -> None:
    logging.info(f"Starting asset staging to {assets_dir}")
    assert os.path.isdir(assets_dir)

    # Fetch datasets largest first, to minimize overall download time
    datasets = sorted(datasets, key=lambda d: d.asset_h5ad_filesize, reverse=True)

    N = len(datasets)
    n_workers = max(min(8, cpu_count()), 64)
    with create_process_pool_executor(args, n_workers) as pe:
        paths = list(
            pe.map(
                copy_file,
                ((n, dataset, assets_dir, N) for n, dataset in enumerate(datasets, start=1)),
            )
        )

    for i in range(len(datasets)):
        datasets[i].dataset_h5ad_path = paths[i]


def _copy_file(n: int, dataset: Dataset, asset_dir: str, N: int) -> str:
    HTTP_GET_TIMEOUT_SEC = 2 * 60 * 60  # just a very big timeout
    protocol = urllib.parse.urlparse(dataset.corpora_asset_h5ad_uri).scheme
    fs = fsspec.filesystem(
        protocol,
        client_kwargs={"timeout": aiohttp.ClientTimeout(total=HTTP_GET_TIMEOUT_SEC, connect=None)},
    )
    dataset_file_name = f"{dataset.dataset_id}.h5ad"
    dataset_path = f"{asset_dir}/{dataset_file_name}"

    logging.info(f"Staging {dataset.dataset_id} ({n} of {N}) to {dataset_path}")
    fs.get_file(dataset.corpora_asset_h5ad_uri, dataset_path)
    logging.info(f"Staging {dataset.dataset_id} ({n} of {N}) complete")
    return dataset_file_name


def copy_file(args: Tuple[int, Dataset, str, int]) -> str:
    return _copy_file(*args)


def cat_file(url: str) -> bytes:
    with fsspec.open(url, compression="infer") as f:
        content = cast(bytes, f.read())  # fsspec has no typing, yet

    return content
