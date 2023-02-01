import argparse
import concurrent.futures
import logging
from typing import List

import tiledbsoma as soma

from .mp import create_process_pool_executor

if soma.get_storage_engine() == "tiledb":
    import tiledb


def consolidate(args: argparse.Namespace, uri: str) -> None:
    """
    This is a non-portable, TileDB-specific consolidation routine.
    """
    if soma.get_storage_engine() != "tiledb":
        return

    census = soma.Collection(uri)
    if not census.exists():
        return

    with create_process_pool_executor(args) as ppe:
        futures = consolidate_collection(args, census, ppe)
    for future in concurrent.futures.as_completed(futures):
        uri = future.result()
        logging.info(f"Consolidate: completed {uri}")


def consolidate_collection(
    args: argparse.Namespace,
    collection: soma.Collection,
    ppe: concurrent.futures.ProcessPoolExecutor,
) -> List[concurrent.futures.Future[str]]:
    futures = []
    for soma_obj in collection.values():
        type = soma_obj.soma_type
        if type in ["SOMADataFrame", "SOMASparseNDArray", "SOMADenseNDArray"]:
            logging.info(f"Consolidate: queuing {type} {soma_obj.uri}")
            futures.append(ppe.submit(consolidate_tiledb_object, soma_obj.uri))
        elif type in ["SOMACollection", "SOMAExperiment", "SOMAMeasurement"]:
            futures += consolidate_collection(args, soma_obj, ppe)
        else:
            raise TypeError(f"Unknown SOMA type {type}.")

    return futures


def consolidate_tiledb_object(uri: str) -> str:
    logging.info(f"Consolidate: starting {uri}")
    tiledb.consolidate(
        uri, config=tiledb.Config({"sm.consolidation.buffer_size": 1 * 1024**3})
    )
    tiledb.vacuum(uri)
    return uri
