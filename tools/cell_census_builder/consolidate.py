import argparse
import concurrent.futures
import logging
from typing import List

import tiledbsoma as soma

from .globals import SOMA_TileDB_Context
from .mp import create_process_pool_executor, log_on_broken_process_pool


def consolidate(args: argparse.Namespace, uri: str) -> None:
    """
    This is a non-portable, TileDB-specific consolidation routine.
    """
    if soma.get_storage_engine() != "tiledb":
        return

    logging.info("Consolidate: started")

    # Gather URIs for any arrays that potentially need consolidation
    with soma.Collection.open(uri, context=SOMA_TileDB_Context()) as census:
        uris_to_consolidate = _walk_tree(census)
    logging.info(f"Consolidate: found {len(uris_to_consolidate)} TileDB objects to consolidate")

    # Queue consolidator for each array
    with create_process_pool_executor(args) as ppe:
        futures = [ppe.submit(consolidate_tiledb_object, uri) for uri in uris_to_consolidate]
        logging.info(f"Consolidate: {len(futures)} consolidation jobs queued")

        # Wait for consolidation to complete
        for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            log_on_broken_process_pool(ppe)
            uri = future.result()
            logging.info(f"Consolidate: completed [{n} of {len(futures)}]: {uri}")

    logging.info("Consolidate: finished")


def _walk_tree(collection: soma.Collection) -> List[str]:
    uris = []
    for soma_obj in collection.values():
        type = soma_obj.soma_type
        if type in ["SOMADataFrame", "SOMASparseNDArray", "SOMADenseNDArray"]:
            uris.append(soma_obj.uri)
        elif type in ["SOMACollection", "SOMAExperiment", "SOMAMeasurement"]:
            uris += _walk_tree(soma_obj)
        else:
            raise TypeError(f"Unknown SOMA type {type}.")
    return uris


def consolidate_tiledb_object(uri: str) -> str:
    assert soma.get_storage_engine() == "tiledb"

    import tiledb

    logging.info(f"Consolidate: start uri {uri}")
    tiledb.consolidate(uri, config=tiledb.Config({"sm.consolidation.buffer_size": 1 * 1024**3}))
    tiledb.vacuum(uri)
    logging.info(f"Consolidate: end uri {uri}")
    return uri
