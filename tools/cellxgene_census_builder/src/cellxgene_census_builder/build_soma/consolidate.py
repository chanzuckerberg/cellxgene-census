import concurrent.futures
import logging
from typing import List

import tiledbsoma as soma

from ..build_state import CensusBuildArgs
from .globals import DEFAULT_TILEDB_CONFIG, SOMA_TileDB_Context
from .mp import create_process_pool_executor, log_on_broken_process_pool


def consolidate(args: CensusBuildArgs, uri: str) -> None:
    """
    This is a non-portable, TileDB-specific consolidation routine.
    """
    if soma.get_storage_engine() != "tiledb":
        return

    logging.info("Consolidate: started")
    uris_to_consolidate = _gather(uri)
    _run(args, uris_to_consolidate)
    logging.info("Consolidate: finished")


def _gather(uri: str) -> List[str]:
    # Gather URIs for any arrays that potentially need consolidation
    with soma.Collection.open(uri, context=SOMA_TileDB_Context()) as census:
        uris_to_consolidate = list_uris_to_consolidate(census)
    logging.info(f"Consolidate: found {len(uris_to_consolidate)} TileDB objects to consolidate")
    return uris_to_consolidate


def _run(args: CensusBuildArgs, uris_to_consolidate: List[str]) -> None:
    # Queue consolidator for each array
    with create_process_pool_executor(args) as ppe:
        futures = [ppe.submit(consolidate_tiledb_object, uri) for uri in uris_to_consolidate]
        logging.info(f"Consolidate: {len(futures)} consolidation jobs queued")

        # Wait for consolidation to complete
        for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            log_on_broken_process_pool(ppe)
            uri = future.result()
            logging.info(f"Consolidate: completed [{n} of {len(futures)}]: {uri}")


def list_uris_to_consolidate(collection: soma.Collection) -> List[str]:
    """
    Recursively walk the soma.Collection and return all uris for soma_types that can be consolidated.
    """
    uris = []
    for soma_obj in collection.values():
        type = soma_obj.soma_type
        if type in ["SOMADataFrame", "SOMASparseNDArray", "SOMADenseNDArray"]:
            uris.append(soma_obj.uri)
        elif type in ["SOMACollection", "SOMAExperiment", "SOMAMeasurement"]:
            uris += list_uris_to_consolidate(soma_obj)
        else:
            raise TypeError(f"Unknown SOMA type {type}.")
    return uris


def consolidate_tiledb_object(uri: str) -> str:
    assert soma.get_storage_engine() == "tiledb"

    import tiledb

    logging.info(f"Consolidate: start uri {uri}")

    for mode in ["fragment_meta", "array_meta", "fragments", "commits"]:
        ctx = tiledb.Ctx(
            tiledb.Config(
                {
                    **DEFAULT_TILEDB_CONFIG,
                    "sm.consolidation.mode": mode,
                    "sm.vacuum.mode": mode,
                }
            )
        )
        tiledb.consolidate(uri, ctx=ctx)
        tiledb.vacuum(uri, ctx=ctx)

    logging.info(f"Consolidate: end uri {uri}")
    return uri
