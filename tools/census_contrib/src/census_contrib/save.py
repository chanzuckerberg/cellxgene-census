from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
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
                    "filters": None,
                }
            },
            "cell_order": "row-major",
            "tile_order": "row-major",
            "allows_duplicates": True,
        },
    }
}


def soma_data_filter(value_range: Tuple[float, float], float_mode: str, sig_bits: int = 20) -> PlatformConfig:
    """Given an embedding's value range, generate appropriate filter pipeline for soma_data attribute"""
    if float_mode == "trunc":
        return [
            "ByteShuffleFilter",
            {"_type": "ZstdFilter", "level": 19},
        ]

    assert float_mode == "scale"

    dmin, dmax = value_range
    if dmin is None or dmax is None or dmin >= dmax:
        raise ValueError("Value range malformed, expected [min,max]")

    offset = dmin
    bytewidth = 4
    factor = 1.0 / ((2 ** (sig_bits - math.log2(dmax - dmin))) - 1)
    assert 0 < sig_bits <= (8 * bytewidth)

    # FloatScaleFilter stores round((raw_float - offset) / factor), as an signed int of width bytewidth
    return [
        {
            "_type": "FloatScaleFilter",
            "factor": factor,
            "offset": offset,
            "bytewidth": bytewidth,
        },
        "BitShuffleFilter",
        {"_type": "ZstdFilter", "level": 5},
    ]


def make_platform_config(shape: Tuple[int, int], value_range: Tuple[float, float], float_mode: str) -> PlatformConfig:
    platform_config = copy.deepcopy(PLATFORM_CONFIG_TEMPLATE)

    tdb_schema = platform_config["tiledb"]["create"]
    tdb_schema["dims"]["soma_dim_1"]["tile"] = shape[1]
    tdb_schema["attrs"]["soma_data"]["filters"] = soma_data_filter(value_range, float_mode)

    return platform_config


def create_obsm_like_array(
    uri: Union[str, Path],
    value_range: Tuple[float, float],  # closed, i.e., inclusive [min, max]
    shape: Tuple[int, int],
    float_mode: str,
    context: Optional[soma.options.SOMATileDBContext] = None,
) -> soma.SparseNDArray:
    """Create and return opened array. Can be used as a context manager."""
    array_path: str = Path(uri).as_posix()
    return soma.SparseNDArray.create(
        array_path,
        type=pa.float32(),
        shape=shape,
        context=context,
        platform_config=make_platform_config(shape, value_range, float_mode),
    )


def apply_float_mode(tbl: pa.Table, float_mode: str, sig_bits: int = 7) -> pa.Table:
    if float_mode == "scale":
        # noop - done w/ TileDB filter
        return tbl

    assert float_mode == "trunc"
    assert tbl.field("d").type == pa.float32()
    d = tbl.column("d").combine_chunks().to_numpy(zero_copy_only=False, writable=True)
    d = roundHalfToEven(d, sig_bits)
    tbl = tbl.set_column(tbl.column_names.index("d"), "d", pa.array(d))
    return tbl


def roundHalfToEven(a: npt.NDArray[np.float32], keepbits: int) -> npt.NDArray[np.float32]:
    """
    Generate reduced precision floating point array, with round half to even.

    IMPORANT: In-place operation.

    Ref: https://gmd.copernicus.org/articles/14/377/2021/gmd-14-377-2021.html
    """
    assert a.dtype == np.float32  # code below assumes this
    nmant = 23
    bits = 32
    if keepbits < 1 or keepbits >= nmant:
        return a
    maskbits = nmant - keepbits
    full_mask = (1 << bits) - 1
    mask = (full_mask >> maskbits) << maskbits
    half_quantum1 = (1 << (maskbits - 1)) - 1

    b = a.view(np.int32)
    b += ((b >> maskbits) & 1) + half_quantum1
    b &= mask
    return a


def consolidate_array(uri: Union[str, Path]) -> None:
    import tiledb

    array_path: str = Path(uri).as_posix()

    for mode in ("fragment_meta", "array_meta", "fragments", "commits"):
        try:
            ctx = tiledb.Ctx(
                tiledb.Config(
                    {
                        "py.init_buffer_bytes": 4 * 1024**3,
                        "soma.init_buffer_bytes": 4 * 1024**3,
                        "sm.consolidation.buffer_size": 4 * 1024**3,
                        "sm.consolidation.mode": mode,
                        "sm.vacuum.mode": mode,
                    }
                )
            )
            logger.info(f"Consolidate: start mode={mode}, uri={array_path}")
            tiledb.consolidate(array_path, ctx=ctx)
            logger.info(f"Vacuum: start mode={mode}, uri={array_path}")
            tiledb.vacuum(array_path, ctx=ctx)
        except tiledb.TileDBError as e:
            logger.error(f"Consolidation error, uri={array_path}: {str(e)}")
            raise

    logger.info(f"Consolidate/vacuum: end uri={array_path}")
