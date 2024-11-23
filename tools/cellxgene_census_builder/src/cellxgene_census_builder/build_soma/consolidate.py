import concurrent.futures
import logging
import re
import time
from collections.abc import Sequence
from typing import overload

import attrs
import dask.distributed
import psutil
import tiledb
import tiledbsoma as soma

from ..util import clamp, cpu_count
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
) -> list[concurrent.futures.Future[str]]: ...


@overload
def submit_consolidate(
    uri: str,
    pool: dask.distributed.Client,
    vacuum: bool,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[dask.distributed.Future]: ...


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
    from tiledbsoma._soma_group import SOMAGroup

    uris = []
    for soma_obj in collection.values():
        if isinstance(soma_obj, SOMAGroup):
            uris += list_uris_to_consolidate(soma_obj)
            n_columns = 0
            n_fragments = 0
        else:
            n_columns = len(soma_obj.schema)
            n_fragments = len(tiledb.array_fragments(soma_obj.uri))
        uris.append(
            ConsolidationCandidate(
                uri=soma_obj.uri, soma_type=soma_obj.soma_type, n_columns=n_columns, n_fragments=n_fragments
            )
        )

    return uris


def _consolidate_array(
    obj: ConsolidationCandidate,
    vacuum: bool,
    consolidation_modes: list[str] | None = None,
    consolidation_config: dict[str, str] | None = None,
) -> None:
    # TODO: Consolidation for dense arrays is currently broken, tracked in https://github.com/single-cell-data/TileDB-SOMA/issues/3383
    if obj.soma_type == "SOMADenseNDArray":
        return

    modes = consolidation_modes or ["fragment_meta", "array_meta", "commits", "fragments"]
    uri = obj.uri

    # use ~1/32th of RAM, clamped to [1, 32].
    total_buffer_size = clamp(int(psutil.virtual_memory().total / 32 // 1024**3), 1, 32) * 1024**3

    for mode in modes:
        tiledb.consolidate(
            uri,
            config=tiledb.Config(
                {
                    **DEFAULT_TILEDB_CONFIG,
                    "sm.consolidation.mode": mode,
                    "sm.consolidation.total_buffer_size": total_buffer_size,
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
