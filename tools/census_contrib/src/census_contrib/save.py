from __future__ import annotations

import copy
import math
import pathlib

import pyarrow as pa
import tiledbsoma as soma
from somacore.options import PlatformConfig

from .args import Arguments
from .embedding import EmbeddingIJD
from .metadata import ContribMetadata
from .util import error, soma_context

PLATFORM_CONFIG: PlatformConfig = {
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


def soma_data_filter(emb: EmbeddingIJD, sig_bits: int = 20) -> PlatformConfig:
    """Given an embedding, generate appropriate filter pipeline for soma_data attribute"""
    d = emb.d
    dmin = d.min()
    dmax = d.max()

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


def save_as_soma(args: Arguments, metadata: ContribMetadata, emb: EmbeddingIJD) -> None:
    save_to = pathlib.PosixPath(args.save_soma_to)
    if save_to.exists():
        error(args, "SOMA output path already exists")

    save_to.mkdir(parents=True, exist_ok=True)

    emb_tbl = pa.Table.from_pydict(
        {
            "soma_dim_0": pa.array(emb.i),
            "soma_dim_1": pa.array(emb.j),
            "soma_data": pa.array(emb.d),
        }
    )

    platform_config = copy.deepcopy(PLATFORM_CONFIG)
    tdb_schema = platform_config["tiledb"]["create"]
    tdb_schema["dims"]["soma_dim_1"]["tile"] = emb.shape[1]
    tdb_schema["attrs"]["soma_data"]["filters"] = soma_data_filter(emb)

    args.logger.info("Creating SOMA array")
    with soma.SparseNDArray.create(
        save_to.as_posix(),
        type=pa.float32(),
        shape=emb.shape,
        context=soma_context(),
        platform_config=platform_config,
    ) as A:
        args.logger.info("Writing SOMA array - starting")
        A.metadata["CxG_accession_id"] = args.accession
        A.metadata["CxG_contrib_metadata"] = metadata.as_json()
        A.write(emb_tbl)
        args.logger.info("Writing SOMA array - completed")
