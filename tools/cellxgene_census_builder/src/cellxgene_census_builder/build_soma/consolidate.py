import logging
import re
from concurrent.futures import Executor, Future, as_completed
from typing import List, Optional, Sequence

import attrs
import tiledb
import tiledbsoma as soma

from ..build_state import CensusBuildArgs
from ..util import cpu_count
from .globals import DEFAULT_TILEDB_CONFIG, SOMA_TileDB_Context
from .mp import create_process_pool_executor, log_on_broken_process_pool

logger = logging.getLogger(__name__)


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
    """The old API - consolidate & vacuum everything, and return when done."""
    with create_process_pool_executor(args) as ppe:
        futures = submit_consolidate(args, uri, pool=ppe, vacuum=True, exclude=None)

        # Wait for consolidation to complete
        for future in as_completed(futures):
            log_on_broken_process_pool(ppe)
            uri = future.result()


def submit_consolidate(
    args: CensusBuildArgs,
    uri: str,
    pool: Executor,
    vacuum: bool,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> Sequence[Future[str]]:
    """
    This is a non-portable, TileDB-specific consolidation routine. Returns sequence of
    futures, each of which returns the URI for the array/group.

    Will vacuum if requested. Excludes any object URI matching a regex in the exclude list.
    """
    if soma.get_storage_engine() != "tiledb":
        return ()

    exclude = [] if exclude is None else exclude
    include = [r".*"] if include is None else include
    uris_to_consolidate = [
        obj
        for obj in _gather(uri)
        if any(re.fullmatch(i, obj.uri) for i in include) and not any(re.fullmatch(e, obj.uri) for e in exclude)
    ]
    logger.info(f"Consolidate: found {len(uris_to_consolidate)} TileDB objects to consolidate")

    futures = [pool.submit(_consolidate_tiledb_object, uri, vacuum) for uri in uris_to_consolidate]
    logger.info(f"Consolidate: {len(futures)} consolidation jobs queued")
    return futures


def _gather(uri: str) -> List[ConsolidationCandidate]:
    # Gather URIs for any arrays that potentially need consolidation
    with soma.Collection.open(uri, context=SOMA_TileDB_Context()) as census:
        uris_to_consolidate = list_uris_to_consolidate(census)
    return uris_to_consolidate


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


def _consolidate_array(uri: str, vacuum: bool) -> None:
    modes = ["fragment_meta", "array_meta", "commits", "fragments"]

    for mode in modes:
        tiledb.consolidate(
            uri,
            config=tiledb.Config(
                {
                    **DEFAULT_TILEDB_CONFIG,
                    "sm.compute_concurrency_level": max(16, cpu_count()),
                    "sm.io_concurrency_level": max(16, cpu_count()),
                    # once we update to TileDB core 2.19, remove this and replace
                    # with sm.consolidation.total_buffer_size
                    "sm.consolidation.buffer_size": 1 * 1024**3,
                    "sm.consolidation.mode": mode,
                }
            ),
        )

        if vacuum:
            tiledb.vacuum(uri, config=tiledb.Config({**DEFAULT_TILEDB_CONFIG, "sm.vacuum.mode": mode}))


def _consolidate_group(uri: str, vacuum: bool) -> None:
    tiledb.Group.consolidate_metadata(uri, config=tiledb.Config({**DEFAULT_TILEDB_CONFIG}))
    if vacuum:
        tiledb.Group.vacuum_metadata(uri, config=tiledb.Config({**DEFAULT_TILEDB_CONFIG}))


def _consolidate_tiledb_object(obj: ConsolidationCandidate, vacuum: bool) -> str:
    assert soma.get_storage_engine() == "tiledb"

    try:
        logger.info(f"Consolidate[vacuum={vacuum}] start, uri={obj.uri}")
        if obj.is_array():
            _consolidate_array(obj.uri, vacuum)
        else:
            _consolidate_group(obj.uri, vacuum)
        logger.info(f"Consolidate[vacuum={vacuum}] finish, uri={obj.uri}")
        return obj.uri
    except tiledb.TileDBError as e:
        logger.error(f"Consolidate[vacuum={vacuum}] error, uri={obj.uri}: {str(e)}")
        raise
