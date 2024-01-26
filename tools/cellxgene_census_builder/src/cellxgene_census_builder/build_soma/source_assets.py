from __future__ import annotations

import copy
import logging
import os
import pathlib
import time
from typing import Any, List, cast

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


def stage_source_assets(datasets: List[Dataset], args: CensusBuildArgs) -> None:
    """NOTE: non-pure -- modifies Datasets in place."""
    assets_dir = args.h5ads_path

    # e.g., "census-builder-prod/1.0.0"
    user_agent = f"{args.config.user_agent_prefix}{args.config.user_agent_environment}/{__version__}"

    logger.info(f"Starting asset staging to {assets_dir}")
    assert os.path.isdir(assets_dir)

    def copy_file(args: tuple[str, str]) -> int:
        """copy from->to, return bytes read"""
        from_path, to_path = args
        HTTP_GET_TIMEOUT_SEC = 2 * 60 * 60  # just a very big timeout
        storage_options = {
            "timeout": aiohttp.ClientTimeout(total=HTTP_GET_TIMEOUT_SEC, connect=None),
            "headers": {"User-Agent": user_agent},
        }
        return pcopyfile(from_path, to_path, exist_ok=True, **storage_options)

    for dataset in datasets:
        dataset.dataset_h5ad_path = f"{dataset.dataset_id}.h5ad"

    bytes_read = (
        dask.bag.from_sequence(
            (d.dataset_asset_h5ad_uri, (assets_dir / d.dataset_h5ad_path).as_posix()) for d in datasets
        )
        .map(copy_file)
        .compute()
    )

    for d, n_bytes in zip(datasets, bytes_read):
        # Confirm expected number of bytes.
        # TODO: add integrity checksum as well (blocked on chanzuckerberg/single-cell-data-portal#4392)
        actual_fsize = os.path.getsize(assets_dir / d.dataset_h5ad_path)
        if d.asset_h5ad_filesize == -1:  # i.e. no prior expectation of size
            d.asset_h5ad_filesize = n_bytes
        assert (
            d.asset_h5ad_filesize == n_bytes == actual_fsize
        ), f"Error reading {d.dataset_id}: got {actual_fsize} bytes, expected {d.asset_h5ad_filesize}"

    return


def pcopyfile(from_url: str, to_path: str, exist_ok: bool = True, block_size: int = 64 * 2**20, **kwargs: Any) -> int:
    """
    Parallel copy of file from_url->to_path. Assumes support for block fetches, a la
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
            (OpenFile(fs, path, compression=compression), offset, length) for offset, length in zip(offsets, lengths)
        )

    def _read_a_block(args: tuple[OpenFile, int, int]) -> tuple[int, bytes]:
        # read block into memory
        filelike, blk_off, blk_len = args
        with copy.copy(filelike) as f:
            sleep_for_secs = 3
            last_error: aiohttp.ClientPayloadError | None = None
            for attempt in range(4):
                try:
                    return (blk_off, read_block(f, blk_off, blk_len))
                except (aiohttp.ClientPayloadError, ConnectionError) as e:
                    logger.error(f"Fetch of {from_url} failed: {str(e)}")
                    last_error = e
                    time.sleep(2**attempt * sleep_for_secs)
            else:
                assert last_error is not None
                raise last_error

    def copy_block(args: tuple[int, Delayed], outfile_name: str) -> int:
        # write block to file
        block_offset, block_data = args

        # write block to out file
        with open(outfile_name, "rb+") as outfile:
            outfile.seek(block_offset)
            cnt = outfile.write(block_data)
            assert cnt == len(block_data)

        return cnt

    delayed_copy_blocks = (
        get_file_blocks(from_url, blocksize=block_size).map(_read_a_block).map(copy_block, outfile_name=to_path)
    )

    total_bytes_written = cast(int, sum(delayed_copy_blocks.compute()))
    logger.debug(f"Copy complete, url={from_url}, bytes={total_bytes_written}")

    return total_bytes_written
