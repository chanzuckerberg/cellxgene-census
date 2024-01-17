import logging
import os
import time
import urllib.parse
from typing import List, Tuple, cast

import aiohttp
import fsspec

from .. import __version__
from ..build_state import CensusBuildArgs
from ..util import cpu_count
from .datasets import Dataset
from .mp import create_process_pool_executor

logger = logging.getLogger(__name__)


def stage_source_assets(datasets: List[Dataset], args: CensusBuildArgs) -> None:
    assets_dir = args.h5ads_path.as_posix()

    # e.g., "census-builder-prod/1.0.0"
    user_agent = f"{args.config.user_agent_prefix}{args.config.user_agent_environment}/{__version__}"

    logger.info(f"Starting asset staging to {assets_dir}")
    assert os.path.isdir(assets_dir)

    # Fetch datasets largest first, to minimize overall download time
    datasets = sorted(datasets, key=lambda d: d.asset_h5ad_filesize, reverse=True)

    N = len(datasets)
    if args.config.multi_process:
        n_workers = min(max(8, cpu_count()), 256)
        with create_process_pool_executor(args, max_workers=n_workers) as pe:
            paths = list(
                pe.map(
                    copy_file, ((n, dataset, assets_dir, N, user_agent) for n, dataset in enumerate(datasets, start=1))
                )
            )
    else:
        paths = [copy_file((n, dataset, assets_dir, N, user_agent)) for n, dataset in enumerate(datasets, start=1)]

    for i in range(len(datasets)):
        datasets[i].dataset_h5ad_path = paths[i]


def _copy_file(n: int, dataset: Dataset, asset_dir: str, N: int, user_agent: str) -> str:
    HTTP_GET_TIMEOUT_SEC = 2 * 60 * 60  # just a very big timeout
    protocol = urllib.parse.urlparse(dataset.dataset_asset_h5ad_uri).scheme
    fs = fsspec.filesystem(
        protocol,
        client_kwargs={
            "timeout": aiohttp.ClientTimeout(total=HTTP_GET_TIMEOUT_SEC, connect=None),
            "headers": {"User-Agent": user_agent},
        },
    )
    dataset_file_name = f"{dataset.dataset_id}.h5ad"
    dataset_path = f"{asset_dir}/{dataset_file_name}"

    logger.info(f"Staging {dataset.dataset_id} ({n} of {N}) to {dataset_path}")

    sleep_for_secs = 10
    last_error: aiohttp.ClientPayloadError | None = None
    for attempt in range(4):
        try:
            fs.get_file(dataset.dataset_asset_h5ad_uri, dataset_path)
            break
        except aiohttp.ClientPayloadError as e:
            logger.error(f"Fetch of {dataset.dataset_id} at {dataset_path} failed: {str(e)}")
            last_error = e
            time.sleep(2**attempt * sleep_for_secs)
    else:
        assert last_error is not None
        raise last_error

    # verify file size is as expected, if we know the size a priori
    assert (dataset.asset_h5ad_filesize == -1) or (
        dataset.asset_h5ad_filesize == os.path.getsize(dataset_path)
    ), f"Download for {dataset.dataset_asset_h5ad_uri} failed - "
    f"expected size {dataset.asset_h5ad_filesize}, "
    f"found size {os.path.getsize(dataset_path)}"
    # TODO: add integrity checksum as well. Waiting on feature request chanzuckerberg/single-cell-data-portal#4392

    logger.info(f"Staging {dataset.dataset_id} ({n} of {N}) complete")
    return dataset_file_name


def copy_file(args: Tuple[int, Dataset, str, int, str]) -> str:
    return _copy_file(*args)


def cat_file(url: str) -> bytes:
    with fsspec.open(url, compression="infer") as f:
        content = cast(bytes, f.read())  # fsspec has no typing, yet

    return content
