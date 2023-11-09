from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import pyarrow as pa
import tiledbsoma as soma
from somacore.options import PlatformConfig

from .util import get_logger

logger = get_logger()


PLATFORM_CONFIG_TEMPLATE: PlatformConfig = {
    "tiledb": {
        "create": {
            "capacity": 2**16,
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
                    "filters": [
                        "ByteShuffleFilter",
                        {"_type": "ZstdFilter", "level": 5},
                    ]
                }
            },
            "cell_order": "row-major",
            "tile_order": "row-major",
            "allows_duplicates": True,
        },
    }
}


def soma_data_filter(value_range: Tuple[float, float], sig_bits: int = 20) -> PlatformConfig:
    """Given an embedding's value range, generate appropriate filter pipeline for soma_data attribute"""
    dmin, dmax = value_range

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


def make_platform_config(shape: Tuple[int, int], value_range: Tuple[float, float]) -> PlatformConfig:
    platform_config = copy.deepcopy(PLATFORM_CONFIG_TEMPLATE)

    tdb_schema = platform_config["tiledb"]["create"]
    tdb_schema["dims"]["soma_dim_1"]["tile"] = shape[1]
    tdb_schema["attrs"]["soma_data"]["filters"] = soma_data_filter(value_range)

    return platform_config


def create_obsm_like_array(
    uri: str,
    value_range: Tuple[float, float],  # closed, i.e., inclusive [min, max]
    shape: Tuple[int, int],
    context: Optional[soma.options.SOMATileDBContext] = None,
) -> soma.SparseNDArray:
    return soma.SparseNDArray.create(
        uri,
        type=pa.float32(),
        shape=shape,
        context=context,
        platform_config=make_platform_config(shape, value_range),
    )


def consolidate_array(uri: str) -> None:
    import tiledb

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
            logger.info(f"Consolidate: start mode={mode}, uri={uri}")
            tiledb.consolidate(uri, ctx=ctx)
            logger.info(f"Vacuum: start mode={mode}, uri={uri}")
            tiledb.vacuum(uri, ctx=ctx)
        except tiledb.TileDBError as e:
            logger.error(f"Consolidation error, uri={uri}: {str(e)}")
            raise

    logger.info(f"Consolidate/vacuum: end uri={uri}")
