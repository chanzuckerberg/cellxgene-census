import pathlib

import pyarrow as pa
import tiledbsoma as soma

from .args import Arguments
from .embedding import EmbeddingIJD
from .metadata import ContribMetadata
from .util import error


PLATFORM_CONFIG = {
    "tiledb": {
        "create": {
            "capacity": 2**16,
            "dims": {
                "soma_dim_0": {
                    "tile": 2048,
                    "filters": [
                        "PositiveDeltaFilter",
                        {"_type": "ZstdFilter", "level": 9},
                    ],
                },
                "soma_dim_1": {
                    "tile": 2048,
                    "filters": [
                        "ByteShuffleFilter",
                        {"_type": "ZstdFilter", "level": 9},
                    ],
                },
            },
            "attrs": {
                "soma_data": {
                    "filters": [
                        # {
                        #     "_type": "FloatScaleFilter",
                        #     "factor": 1.0 / 100_000,  # scaling
                        #     # offset: ?,
                        #     "bytewidth": 4,
                        # },
                        "ByteShuffleFilter",
                        {"_type": "ZstdFilter", "level": 9},
                    ]
                }
            },
            "cell_order": "row-major",
            "tile_order": "row-major",
            "allows_duplicates": True,
        },
    }
}


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

    args.logger.info("Creating SOMA array")
    with soma.SparseNDArray.create(
        save_to.as_posix(),
        type=pa.float32(),
        shape=emb.shape,
        platform_config=PLATFORM_CONFIG,
    ) as A:
        args.logger.info("Writing SOMA array")
        A.metadata["CxG_accession_id"] = args.accession
        A.metadata["CxG_contrib_metadata"] = metadata.as_json()
        A.write(emb_tbl)
