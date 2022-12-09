import argparse
import concurrent.futures
import logging

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

    consolidate_collection(args, census)


def consolidate_collection(args: argparse.Namespace, collection: soma.Collection) -> None:

    futures = []
    with create_process_pool_executor(args, max_workers=4) as ppe:
        for soma_obj in collection.values():
            type = soma_obj.soma_type
            if type in ["SOMADataFrame", "SOMASparseNdArray", "SOMADenseNdArray"]:
                logging.info(f"Consolidate: starting {type} {soma_obj.uri}")
                futures.append(ppe.submit(consolidate_tiledb_object, soma_obj.uri))
            elif type in ["SOMACollection", "SOMAExperiment", "SOMAMeasurement"]:
                consolidate_collection(args, soma_obj)
            else:
                raise TypeError(f"Unknown SOMA type {type}.")

        for future in concurrent.futures.as_completed(futures):
            uri = future.result()
            logging.info(f"Consolidate: completed {uri}")


def consolidate_tiledb_object(uri: str) -> str:
    tiledb.consolidate(uri, config=tiledb.Config({"sm.consolidation.buffer_size": 1 * 1024**3}))
    tiledb.vacuum(uri)
    return uri
