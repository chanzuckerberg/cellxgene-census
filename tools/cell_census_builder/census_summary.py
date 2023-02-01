import logging
from typing import Sequence

import pandas as pd
import pyarrow as pa
import tiledbsoma as soma

from .experiment_builder import ExperimentBuilder, get_summary_stats
from .globals import CENSUS_SCHEMA_VERSION, CENSUS_SUMMARY_NAME, SOMA_TileDB_Context
from .util import uricat


def create_census_summary(
    info_collection: soma.Collection,
    experiment_builders: Sequence[ExperimentBuilder],
    build_tag: str,
) -> None:
    logging.info("Creating census summary")

    summary_stats = get_summary_stats(experiment_builders)
    data = [
        ("cell_census_schema_version", CENSUS_SCHEMA_VERSION),
        ("cell_census_build_date", build_tag),
        ("total_cell_count", str(summary_stats["total_cell_count"])),
        ("unique_cell_count", str(summary_stats["unique_cell_count"])),
        (
            "number_donors_homo_sapiens",
            str(summary_stats["number_donors"]["homo_sapiens"]),
        ),
        (
            "number_donors_mus_musculus",
            str(summary_stats["number_donors"]["mus_musculus"]),
        ),
    ]

    df = pd.DataFrame.from_records(data, columns=["label", "value"])
    df["soma_joinid"] = range(len(df))

    # write to a SOMA dataframe
    summary_uri = uricat(info_collection.uri, CENSUS_SUMMARY_NAME)
    summary = soma.DataFrame(summary_uri, context=SOMA_TileDB_Context())
    summary.create(
        pa.Schema.from_pandas(df, preserve_index=False),
        index_column_names=["soma_joinid"],
    )
    for batch in pa.Table.from_pandas(df, preserve_index=False).to_batches():
        summary.write(batch)
    info_collection.set(CENSUS_SUMMARY_NAME, summary, relative=True)
