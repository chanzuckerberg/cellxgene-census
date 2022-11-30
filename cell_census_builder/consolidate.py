import logging

import tiledbsoma as soma

if soma.get_storage_engine() == "tiledb":
    import tiledb


def consolidate(uri: str) -> None:
    """
    This is a non-portable, TileDB-specific consolidation routine.
    """
    if soma.get_storage_engine() != "tiledb":
        return

    census = soma.Collection(uri)
    if not census.exists():
        return

    consolidate_collection(census)


def consolidate_collection(collection: soma.Collection) -> None:
    for soma_obj in collection.values():
        type = soma_obj.soma_type
        if type in ["SOMADataFrame", "SOMASparseNdArray", "SOMADenseNdArray"]:
            logging.info(f"Consolidating {type} {soma_obj.uri}")
            consolidate_tiledb_object(soma_obj.uri)
        elif type in ["SOMACollection", "SOMAExperiment", "SOMAMeasurement"]:
            consolidate_collection(soma_obj)
        else:
            raise TypeError(f"Unknown SOMA type {type}.")


def consolidate_tiledb_object(uri: str) -> None:
    tiledb.consolidate(uri, config=tiledb.Config({"sm.consolidation.buffer_size": 1 * 1024**3}))
    tiledb.vacuum(uri)
