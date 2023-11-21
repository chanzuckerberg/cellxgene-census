from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
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
                        "PositiveDeltaFilter",
                        {"_type": "ZstdFilter", "level": 5},
                    ],
                },
                "soma_dim_1": {
                    "tile": 2048,
                    "filters": [
                        "ByteShuffleFilter",
                        {"_type": "ZstdFilter", "level": 5},
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
            {"_type": "ZstdFilter", "level": 13},
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
        return tbl

    assert float_mode == "trunc"
    assert tbl.field("d").type == pa.float32()

    # IEEE 754 float32 has 23 bit mantissa but look it up just for fun
    nmant = np.finfo(np.float32).nmant
    assert 4 < sig_bits <= nmant
    mask = ~((1 << (nmant - sig_bits)) - 1) & 0xFFFFFFFF
    if sig_bits == nmant:
        return tbl

    tbl = tbl.combine_chunks()
    d = tbl.column("d").chunk(0)
    d = pa.compute.bit_wise_and(d.view(pa.uint32()), pa.scalar(mask, type=pa.uint32())).view(pa.float32())
    tbl = tbl.set_column(tbl.column_names.index("d"), "d", d)
    return tbl


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
