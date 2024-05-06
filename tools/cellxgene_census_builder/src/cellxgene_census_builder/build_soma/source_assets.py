from __future__ import annotations

import copy
import logging
import os
import pathlib
import time
from typing import Any

import aiohttp
import dask
from dask.bag import Bag
from dask.delayed import Delayed
from fsspec.core import OpenFile, get_fs_token_paths
from fsspec.utils import infer_compression, read_block

from .. import __version__
from ..build_state import CensusBuildArgs
from .datasets import Dataset

logger = logging.getLogger(__name__)


def stage_source_assets(datasets: list[Dataset], args: CensusBuildArgs) -> None:
    """NOTE: non-pure -- modifies Datasets argument in place."""
    assets_dir = args.h5ads_path

    # e.g., "census-builder-prod/1.0.0"
    user_agent = f"{args.config.user_agent_prefix}{args.config.user_agent_environment}/{__version__}"
    HTTP_GET_TIMEOUT_SEC = 2 * 60 * 60  # just a very big timeout

    logger.info(f"Starting asset staging to {assets_dir}")
    assert os.path.isdir(assets_dir)

    for dataset in datasets:
        dataset.dataset_h5ad_path = f"{dataset.dataset_id}.h5ad"

    bytes_read = dask.bag.from_delayed(
        [
            dask.delayed(
                (
                    (
                        d,
                        pcopyfile(
                            d.dataset_asset_h5ad_uri,
                            (assets_dir / d.dataset_h5ad_path).as_posix(),
                            exist_ok=True,
                            timeout=aiohttp.ClientTimeout(total=HTTP_GET_TIMEOUT_SEC, connect=None),
                            headers={"User-Agent": user_agent},
                        ),
                    ),
                ),
            )
            for d in datasets
        ]
    ).compute()

    for d, n_bytes in bytes_read:
        # Confirm expected number of bytes.
        # TODO: add integrity checksum as well (blocked on chanzuckerberg/single-cell-data-portal#4392)
        actual_fsize = os.path.getsize(assets_dir / d.dataset_h5ad_path)
        if d.asset_h5ad_filesize == -1:  # i.e. no prior expectation of size
            d.asset_h5ad_filesize = n_bytes
        if not (d.asset_h5ad_filesize == n_bytes == actual_fsize):
            raise ValueError(
                f"Error reading {d.dataset_id}: got {actual_fsize} bytes, expected {d.asset_h5ad_filesize}"
            )


def pcopyfile(
    from_url: str, to_path: str, exist_ok: bool = True, block_size: int = 64 * 2**20, **kwargs: Any
) -> Delayed[int]:
    """Parallel copy of file from_url->to_path. Assumes support for block fetches, a la
    HTTP, S3, etc. Blocks fetched in parallel in no guaranteed order.

    Uses fsspec under the covers. Any additional kwargs are passed to the fsspec session setup,
    and are usually HTTP headers and the like.

    Returns bytes processed (in success case, file size).
    """

    def get_file_blocks(urlpath: str, blocksize: int) -> Bag:
        fs, _, paths = get_fs_token_paths(from_url, mode="rb", storage_options=kwargs)
        if len(paths) != 1:
            raise OSError(f"{from_url} resolved to unexpected number of files")

        path = paths[0]
        size = fs.info(path)["size"]
        if size is None:
            raise ValueError(f"Cannot determine size of {from_url}")

        pathlib.Path(to_path).touch(exist_ok=exist_ok)  # create file if it doesn't exist
        os.truncate(to_path, 0)  # truncate file

        # if file is zero length, do nothing
        if size == 0:
            return dask.bag.from_sequence(())

        offsets = list(range(0, size, blocksize))
        lengths = [blocksize] * len(offsets)
        lengths[-1] = size % blocksize
        compression = infer_compression(path)

        return dask.bag.from_sequence(
            (OpenFile(fs, path, compression=compression), offset, length)
            for offset, length in zip(offsets, lengths, strict=False)
        )

    def _read_a_block(filelike: OpenFile, blk_off: int, blk_len: int) -> tuple[int, bytes]:
        # read block into memory
        with copy.copy(filelike) as f:
            sleep_for_secs = 3
            last_error: aiohttp.ClientPayloadError | aiohttp.ClientResponseError | None = None
            for attempt in range(4):
                try:
                    return (blk_off, read_block(f, blk_off, blk_len))
                except (aiohttp.ClientPayloadError, aiohttp.ClientResponseError, ConnectionError) as e:
                    logger.error(f"Fetch of {from_url} failed: {str(e)}")
                    last_error = e
                    time.sleep(2**attempt * sleep_for_secs)
            else:
                assert last_error is not None
                raise last_error

    def copy_block(block_offset: int, block_data: Delayed, outfile_name: str) -> int:
        # write block to file
        with open(outfile_name, "rb+") as outfile:
            outfile.seek(block_offset)
            cnt = outfile.write(block_data)
            assert cnt == len(block_data)

        return cnt

    delayed_copy_blocks = (
        dask.delayed(get_file_blocks)(from_url, blocksize=block_size)
        .starmap(_read_a_block)
        .starmap(copy_block, outfile_name=to_path)
    )

    @dask.delayed  # type: ignore[misc]
    def _logit(bytes_read: int) -> int:
        logger.debug(f"Copy complete, url={from_url}, bytes={bytes_read}")
        return bytes_read

    return _logit(dask.delayed(sum)(delayed_copy_blocks))
