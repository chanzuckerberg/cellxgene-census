import concurrent.futures
import logging
import re
import threading
import time
from collections.abc import Sequence
from typing import Literal, overload

import attrs
import dask.distributed
import tiledb
import tiledbsoma as soma

from ..util import cpu_count
from .globals import DEFAULT_TILEDB_CONFIG, SOMA_TileDB_Context

logger = logging.getLogger(__name__)


@attrs.define(kw_only=True, frozen=True)
class ConsolidationCandidate:
    uri: str
    soma_type: str
    n_columns: int
    n_fragments: int  # zero if Group

    def is_array(self) -> bool:
        return self.soma_type in [
            "SOMADataFrame",
            "SOMASparseNDArray",
            "SOMADenseNDArray",
        ]

    def is_group(self) -> bool:
        return not self.is_array()


def consolidate_all(uri: str) -> None:
    """Consolidate & vacuum everything, and return when done."""
    for candidate in _gather(uri):
        _consolidate_tiledb_object(candidate, vacuum=True)


@overload
def submit_consolidate(
    uri: str,
    pool: concurrent.futures.ProcessPoolExecutor,
    vacuum: bool,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[concurrent.futures.Future[str]]:
    ...


@overload
def submit_consolidate(
    uri: str,
    pool: dask.distributed.Client,
    vacuum: bool,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[dask.distributed.Future]:
    ...


def submit_consolidate(
    uri: str,
    pool: dask.distributed.Client | concurrent.futures.ProcessPoolExecutor,
    vacuum: bool,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[dask.distributed.Future | concurrent.futures.Future[str]]:
    """This is a non-portable, TileDB-specific consolidation routine. Returns sequence of
    futures, each of which returns the URI for the array/group.

    Will vacuum if requested. Excludes any object URI matching a regex in the exclude list.
    """
    if soma.get_storage_engine() != "tiledb":
        return []

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


def _gather(uri: str) -> list[ConsolidationCandidate]:
    # Gather URIs for any arrays that potentially need consolidation
    with soma.Collection.open(uri, context=SOMA_TileDB_Context()) as census:
        uris_to_consolidate = list_uris_to_consolidate(census)
    return uris_to_consolidate


def list_uris_to_consolidate(
    collection: soma.Collection,
) -> list[ConsolidationCandidate]:
    """Recursively walk the soma.Collection and return all uris for soma_types that can be consolidated and vacuumed."""
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
            n_fragments = 0
        else:
            n_columns = len(soma_obj.schema)
            n_fragments = len(tiledb.array_fragments(soma_obj.uri))
        uris.append(
            ConsolidationCandidate(uri=soma_obj.uri, soma_type=type, n_columns=n_columns, n_fragments=n_fragments)
        )

    return uris


def _consolidate_array(
    obj: ConsolidationCandidate,
    vacuum: bool,
    consolidation_modes: list[str] | None = None,
    consolidation_config: dict[str, str] | None = None,
) -> None:
    modes = consolidation_modes or ["fragment_meta", "array_meta", "commits", "fragments"]
    uri = obj.uri
    for mode in modes:
        tiledb.consolidate(
            uri,
            config=tiledb.Config(
                {
                    **DEFAULT_TILEDB_CONFIG,
                    "sm.consolidation.mode": mode,
                    "sm.consolidation.total_buffer_size": 32 * 1024**3,
                    "sm.compute_concurrency_level": cpu_count(),
                    "sm.io_concurrency_level": cpu_count(),
                    **(consolidation_config or {}),
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


def _consolidate_tiledb_object(
    obj: ConsolidationCandidate,
    vacuum: bool,
    consolidation_modes: list[str] | None = None,
    consolidation_config: dict[str, str] | None = None,
) -> str:
    assert soma.get_storage_engine() == "tiledb"

    try:
        logger.info(f"Consolidate[vacuum={vacuum}] start, uri={obj.uri}")
        consolidate_start_time = time.perf_counter()
        if obj.is_array():
            _consolidate_array(
                obj, vacuum=vacuum, consolidation_modes=consolidation_modes, consolidation_config=consolidation_config
            )
        else:
            _consolidate_group(obj, vacuum=vacuum)
        consolidate_time = time.perf_counter() - consolidate_start_time
        logger.info(f"Consolidate[vacuum={vacuum}] finish, {consolidate_time:.2f} seconds, uri={obj.uri}")
        return obj.uri
    except tiledb.TileDBError as e:
        logger.error(f"Consolidate[vacuum={vacuum}] error, uri={obj.uri}: {str(e)}")
        raise


StopFlag = dict[Literal["stop"], bool]


@attrs.define(frozen=True, kw_only=True)
class AsyncConsolidator:
    stop_request: StopFlag
    thread: threading.Thread


def start_async_consolidation(
    uri: str, fragment_count_threshold: int = 4, polling_period_sec: float = 15.0
) -> AsyncConsolidator:
    """Start an async consolidation process that will safely work alongside writers.
    Intended use is background process during writing of X layers, to reduce total
    fragment count.

    Stop the consolidator by calling `stop_async_consolidation`
    """
    assert fragment_count_threshold > 1
    assert polling_period_sec > 0.0

    stop_request: StopFlag = {"stop": False}
    t = threading.Thread(
        target=_async_consolidator,
        args=(uri, fragment_count_threshold, polling_period_sec, stop_request),
        daemon=True,
        name="Async consolidator",
    )
    t.start()
    return AsyncConsolidator(thread=t, stop_request=stop_request)


def stop_async_consolidation(ac: AsyncConsolidator, *, join: bool = True) -> None:
    """Stop the async consolidator. Will block until it is done."""
    logger.info("Async consolidator: asking for stop")
    ac.stop_request["stop"] = True
    if join:
        ac.thread.join()


def _async_consolidator(
    uri: str, fragment_count_threshold: int, polling_period_sec: float, stop_request: StopFlag
) -> None:
    """Inner loop of async/incremental consolidator."""
    logger.info(f"Async consolidator - starting for {uri}")
    while not stop_request["stop"]:
        # Arrays are the only resource intensive consolidation step. Prioritize the array
        # with the largest number of fragments.
        candidates = sorted(
            (c for c in _gather(uri) if c.is_array() and c.n_fragments > 1),
            key=lambda c: c.n_fragments,
            reverse=True,
        )

        start_time = time.perf_counter()
        for c in candidates:
            if stop_request["stop"]:
                break  # type: ignore[unreachable]

            logger.info(f"Async consolidator: fragments={c.n_fragments}, uri={c.uri}")
            _consolidate_tiledb_object(
                c,
                vacuum=True,
                consolidation_modes=["fragments"],
                consolidation_config={
                    # config for small, incremental consolidation steps.
                    "sm.consolidation.steps": "2",
                    "sm.consolidation.step_min_frags": "2",
                    "sm.consolidation.step_max_frags": "8",
                    "sm.consolidation.max_fragment_size": str(2 * 1024**3),
                },
            )
        else:
            if (time.perf_counter() - start_time) < polling_period_sec:
                time.sleep(polling_period_sec)

    logger.info(f"Async consolidator - stopping for {uri}")
