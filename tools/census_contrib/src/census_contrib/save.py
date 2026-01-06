from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import numpy.typing as npt
import psutil
import pyarrow as pa
import tiledbsoma as soma
from somacore.options import PlatformConfig

from .util import get_logger

logger = get_logger()


PLATFORM_CONFIG_TEMPLATE: PlatformConfig = {
    "tiledb": {
        "create": {
            "capacity": 2**17,
            "dims": {
                "soma_dim_0": {
                    "tile": 2048,
                    "filters": [
                        {"_type": "PositiveDeltaFilter", "window": 2048},
                        {"_type": "ZstdFilter", "level": 9},
                    ],
                },
                "soma_dim_1": {
                    "tile": 2048,
                    "filters": [
                        "DeltaFilter",
                        {"_type": "ZstdFilter", "level": 9},
                    ],
                },
            },
            "attrs": {
                "soma_data": {
                    "filters": [
                        "ByteShuffleFilter",
                        {"_type": "ZstdFilter", "level": 19},
                    ],
                }
            },
            "cell_order": "row-major",
            "tile_order": "row-major",
            "allows_duplicates": True,
        },
    }
}


def make_platform_config(shape: tuple[int, int], value_range: tuple[float, float]) -> PlatformConfig:
    platform_config = copy.deepcopy(PLATFORM_CONFIG_TEMPLATE)
    tdb_schema = platform_config["tiledb"]["create"]
    tdb_schema["dims"]["soma_dim_1"]["tile"] = shape[1]
    return platform_config


def create_sparse_array(
    uri: str | Path,
    value_range: tuple[float, float],  # closed, i.e., inclusive [min, max]
    shape: tuple[int, int],
    context: soma.options.SOMATileDBContext | None = None,
) -> soma.SparseNDArray:
    """Create and return opened sparse nd array. Can be used as a context manager."""
    array_path: str = Path(uri).as_posix()
    return soma.SparseNDArray.create(
        array_path,
        type=pa.float32(),
        shape=shape,
        context=context,
        platform_config=make_platform_config(shape, value_range),
    )


def reduce_float_precision(tbl: pa.Table, sig_bits: int = 7) -> pa.Table:
    assert tbl.field("d").type == pa.float32()
    d = tbl.column("d").combine_chunks().to_numpy(zero_copy_only=False, writable=True)
    d = roundHalfToEven(d, sig_bits)
    tbl = tbl.set_column(tbl.column_names.index("d"), "d", pa.array(d))
    return tbl


def roundHalfToEven(a: npt.NDArray[np.float32], keepbits: int) -> npt.NDArray[np.float32]:
    """Generate reduced precision floating point array, with round half to even.

    IMPORANT: In-place operation.

    Ref: https://gmd.copernicus.org/articles/14/377/2021/gmd-14-377-2021.html
    """
    assert a.dtype == np.float32  # code below assumes IEEE 754 float32
    nmant = 23
    bits = 32
    if keepbits < 1 or keepbits >= nmant:
        return a
    maskbits = nmant - keepbits
    full_mask = (1 << bits) - 1
    mask = (full_mask >> maskbits) << maskbits
    half_quantum1 = np.uint32((1 << (maskbits - 1)) - 1)

    b = a.view(np.uint32)
    b += np.uint32(((b >> maskbits) & 1) + half_quantum1)
    b &= np.uint32(mask)
    return a


def _consolidate_tiledb_object(uri: str | Path, modes: tuple[str, ...]) -> None:
    import tiledb

    cpu_count = psutil.cpu_count() or 1
    sys_mem_GiB = psutil.virtual_memory().total // (1024**3)
    mem_budget_bytes = max(10, min(32, sys_mem_GiB // 8)) * 1024**3

    path: str = Path(uri).as_posix()
    for mode in modes:
        try:
            ctx = tiledb.Ctx(
                tiledb.Config(
                    {
                        "py.init_buffer_bytes": 4 * 1024**3,
                        "soma.init_buffer_bytes": 4 * 1024**3,
                        "sm.mem.total_budget": mem_budget_bytes,
                        "sm.compute_concurrency_level": cpu_count,
                        "sm.io_concurrency_level": cpu_count,
                        "sm.consolidation.mode": mode,
                        "sm.vacuum.mode": mode,
                    }
                )
            )
            logger.info(f"Consolidate: start mode={mode}, uri={path}")
            tiledb.consolidate(path, ctx=ctx)
            logger.info(f"Vacuum: start mode={mode}, uri={path}")
            tiledb.vacuum(path, ctx=ctx)
        except tiledb.TileDBError as e:
            logger.error(f"Consolidation error, uri={path}: {str(e)}")
            raise

    logger.info(f"Consolidate/vacuum: end uri={path}")


def consolidate_array(uri: str | Path) -> None:
    _consolidate_tiledb_object(uri, ("fragment_meta", "array_meta", "fragments", "commits"))


def consolidate_group(uri: str | Path) -> None:
    # TODO: There is a bug in TileDB-Py that prevents consolidation of
    # group metadata. Skipping this step for now - remove this work-around
    # when the bug is fixed. As of 0.23.0, it is not yet fixed.
    #
    # modes = ("group_meta",)
    modes = ()
    _consolidate_tiledb_object(uri, modes)
