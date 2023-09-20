import concurrent.futures
import logging
from typing import List

import attrs
import tiledbsoma as soma

from ..build_state import CensusBuildArgs
from .globals import DEFAULT_TILEDB_CONFIG, SOMA_TileDB_Context
from .mp import create_process_pool_executor, log_on_broken_process_pool


@attrs.define(kw_only=True, frozen=True)
class ConsolidationCandidate:
    uri: str
    soma_type: str

    def is_array(self) -> bool:
        return self.soma_type in [
            "SOMADataFrame",
            "SOMASparseNDArray",
            "SOMADenseNDArray",
        ]

    def is_group(self) -> bool:
        return not self.is_array()


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


def _gather(uri: str) -> List[ConsolidationCandidate]:
    # Gather URIs for any arrays that potentially need consolidation
    with soma.Collection.open(uri, context=SOMA_TileDB_Context()) as census:
        uris_to_consolidate = list_uris_to_consolidate(census)
    logging.info(f"Consolidate: found {len(uris_to_consolidate)} TileDB objects to consolidate")
    return uris_to_consolidate


def _run(args: CensusBuildArgs, uris_to_consolidate: List[ConsolidationCandidate]) -> None:
    # Queue consolidator for each array
    with create_process_pool_executor(args) as ppe:
        futures = [ppe.submit(consolidate_tiledb_object, uri) for uri in uris_to_consolidate]
        logging.info(f"Consolidate: {len(futures)} consolidation jobs queued")

        # Wait for consolidation to complete
        for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            log_on_broken_process_pool(ppe)
            uri = future.result()
            logging.info(f"Consolidate: completed [{n} of {len(futures)}]: {uri}")


def list_uris_to_consolidate(
    collection: soma.Collection,
) -> List[ConsolidationCandidate]:
    """
    Recursively walk the soma.Collection and return all uris for soma_types that can be consolidated and vacuumed.
    """
    uris = []
    for soma_obj in collection.values():
        type = soma_obj.soma_type
        if type not in [
            "SOMACollection",
            "SOMAExperiment",
            "SOMAMeasurement",
            "SOMADataFrame",
            "SOMASparseNDArray",
            "SOMADenseNDArray",
        ]:
            raise TypeError(f"Unknown SOMA type {type}.")

        if type in ["SOMACollection", "SOMAExperiment", "SOMAMeasurement"]:
            uris += list_uris_to_consolidate(soma_obj)
        uris.append(ConsolidationCandidate(uri=soma_obj.uri, soma_type=type))

    return uris


def consolidate_tiledb_object(obj: ConsolidationCandidate) -> str:
    assert soma.get_storage_engine() == "tiledb"

    import tiledb

    uri = obj.uri
    if obj.is_array():
        modes = ["fragment_meta", "array_meta", "fragments", "commits"]
    else:
        # TODO: There is a bug in TileDB-Py that prevents consolidation of
        # group metadata. Skipping this step for now - remove this work-around
        # when the bug is fixed. As of 0.23.0, it is not yet fixed.
        #
        # modes = ["group_meta"]
        modes = []

    # Possible future enhancement - cap fragment size. Increases number of
    # fragments, with unknown perf hit, but could make some ops simpler.
    # For example, this caps each fragment at approximately 10GiB.
    #   "sm.consolidation.max_fragment_size": 10 * 1024**3,

    for mode in modes:
        try:
            ctx = tiledb.Ctx(
                tiledb.Config(
                    {
                        **DEFAULT_TILEDB_CONFIG,
                        "sm.consolidation.buffer_size": 3 * 1024**3,
                        "sm.consolidation.mode": mode,
                        "sm.vacuum.mode": mode,
                    }
                )
            )
            logging.info(f"Consolidate: start mode={mode}, uri={uri}")
            tiledb.consolidate(uri, ctx=ctx)
            logging.info(f"Vacuum: start mode={mode}, uri={uri}")
            tiledb.vacuum(uri, ctx=ctx)
        except tiledb.TileDBError as e:
            logging.error(f"Consolidation error, uri={uri}: {str(e)}")
            raise

    logging.info(f"Consolidate/vacuum: end uri={uri}")
    return uri
