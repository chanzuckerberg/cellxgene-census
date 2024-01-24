import logging
import re
import time
from typing import List, Optional, Sequence

import attrs
import dask.distributed
import tiledb
import tiledbsoma as soma
from dask.distributed import Client, Future

from ..build_state import CensusBuildArgs
from .globals import DEFAULT_TILEDB_CONFIG, SOMA_TileDB_Context

logger = logging.getLogger(__name__)


@attrs.define(kw_only=True, frozen=True)
class ConsolidationCandidate:
    uri: str
    soma_type: str
    n_columns: int

    def is_array(self) -> bool:
        return self.soma_type in [
            "SOMADataFrame",
            "SOMASparseNDArray",
            "SOMADenseNDArray",
        ]

    def is_group(self) -> bool:
        return not self.is_array()


def consolidate(_: CensusBuildArgs, uri: str) -> None:
    """consolidate & vacuum everything, wait for completion"""
    client = dask.distributed.Client.current()
    client.gather(submit_consolidate(uri, pool=client, vacuum=True, exclude=None))


def submit_consolidate(
    uri: str,
    pool: Client,
    vacuum: bool,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> Sequence[Future]:
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
            n_columns = 0
        else:
            n_columns = len(soma_obj.schema)
        uris.append(ConsolidationCandidate(uri=soma_obj.uri, soma_type=type, n_columns=n_columns))

    return uris


def _consolidate_array(obj: ConsolidationCandidate, vacuum: bool) -> None:
    modes = ["fragment_meta", "array_meta", "commits", "fragments"]
    uri = obj.uri
    for mode in modes:
        # Once we update to TileDB core 2.19, remove this and replace with
        # sm.consolidation.total_buffer_size
        #
        # Heuristic based on the fact that our sparse nd arrays (X/*) are both large and
        # have small number of columns. If a low-column count, use bigger buffers.
        buffer_size = 1 * 1024**3 if obj.n_columns > 3 else 4 * 1024**3

        tiledb.consolidate(
            uri,
            config=tiledb.Config(
                {
                    **DEFAULT_TILEDB_CONFIG,
                    # once we update to TileDB core 2.19, remove this and replace
                    # with sm.consolidation.total_buffer_size
                    "sm.consolidation.buffer_size": buffer_size,
                    "sm.consolidation.mode": mode,
                }
            ),
        )

        if vacuum:
            tiledb.vacuum(uri, config=tiledb.Config({**DEFAULT_TILEDB_CONFIG, "sm.vacuum.mode": mode}))


def _consolidate_group(obj: ConsolidationCandidate, vacuum: bool) -> None:
    uri = obj.uri
    tiledb.Group.consolidate_metadata(uri, config=tiledb.Config({**DEFAULT_TILEDB_CONFIG}))
    if vacuum:
        tiledb.Group.vacuum_metadata(uri, config=tiledb.Config({**DEFAULT_TILEDB_CONFIG}))


def _consolidate_tiledb_object(obj: ConsolidationCandidate, vacuum: bool) -> str:
    assert soma.get_storage_engine() == "tiledb"

    try:
        logger.info(f"Consolidate[vacuum={vacuum}] start, uri={obj.uri}")
        consolidate_start_time = time.perf_counter()
        if obj.is_array():
            _consolidate_array(obj, vacuum)
        else:
            _consolidate_group(obj, vacuum)
        consolidate_time = time.perf_counter() - consolidate_start_time
        logger.info(f"Consolidate[vacuum={vacuum}] finish, {consolidate_time:.2f} seconds, uri={obj.uri}")
        return obj.uri
    except tiledb.TileDBError as e:
        logger.error(f"Consolidate[vacuum={vacuum}] error, uri={obj.uri}: {str(e)}")
        raise


def start_async_consolidation(
    client: dask.distributed.Client, uri: str, fragment_count_threshold: int = 4, polling_period_sec: float = 10.0
) -> Future:
    """
    Start an async consolidation process that will safely work alongside writers.
    Intended use is background process during writing of X layers, to reduce total
    fragment count.

    Stop the consolidator by calling `stop_async_consolidation`
    """
    assert fragment_count_threshold > 1
    assert polling_period_sec > 0.0
    return client.submit(_async_consolidator, uri, fragment_count_threshold, polling_period_sec)


def _async_consolidator(uri: str, fragment_count_threshold: int, polling_period_sec: float) -> None:
    logger.info(f"Async consolidator - starting for {uri}")
    consolidation_stop_flag = dask.distributed.Variable("consolidation-manager-stop")
    consolidation_stop_flag.set(False)
    dask.distributed.secede()  # don't consume a worker slot

    while True:
        # Arrays are the only time consuming consolidation step, so focus on them
        candidates = [c for c in _gather(uri) if c.is_array()]
        for candidate in candidates:
            # stop if asked
            if consolidation_stop_flag.get():
                logger.info(f"Async consolidator - stopping for {uri}")
                return

            n_fragments = len(tiledb.array_fragments(candidate.uri))
            if n_fragments > fragment_count_threshold:
                logger.info(f"Async consolidator: fragments={n_fragments}, uri={candidate.uri}")
                _consolidate_tiledb_object(candidate, vacuum=True)

        time.sleep(polling_period_sec)


def stop_async_consolidation(fs: Future) -> None:
    logger.info("Async consolidator: asking for stop")
    consolidation_stop_flag = dask.distributed.Variable("consolidation-manager-stop", client=fs.client)
    consolidation_stop_flag.set(True)
    dask.distributed.wait([fs])
