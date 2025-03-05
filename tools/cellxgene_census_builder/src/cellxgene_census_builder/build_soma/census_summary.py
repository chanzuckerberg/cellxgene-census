import logging
from collections.abc import Sequence

import pandas as pd
import pyarrow as pa
import tiledbsoma as soma

from .experiment_builder import ExperimentBuilder, get_summary_stats
from .globals import CENSUS_INFO_ORGANISMS_NAME, CENSUS_SCHEMA_VERSION, CENSUS_SUMMARY_NAME, CXG_SCHEMA_VERSION

logger = logging.getLogger(__name__)


def create_census_summary(
    info_collection: soma.Collection, experiment_builders: Sequence[ExperimentBuilder], build_tag: str
) -> None:
    logger.info("Creating census summary")

    summary_stats = get_summary_stats(experiment_builders)
    data = [
        ("census_schema_version", CENSUS_SCHEMA_VERSION),
        ("census_build_date", build_tag),
        ("dataset_schema_version", CXG_SCHEMA_VERSION),
        ("total_cell_count", str(summary_stats["total_cell_count"])),
        ("unique_cell_count", str(summary_stats["unique_cell_count"])),
        ("number_donors_homo_sapiens", str(summary_stats["number_donors"]["homo_sapiens"])),
        ("number_donors_mus_musculus", str(summary_stats["number_donors"]["mus_musculus"])),
    ]

    df = pd.DataFrame.from_records(data, columns=["label", "value"])
    df["soma_joinid"] = range(len(df))

    # write to a SOMA dataframe
    with info_collection.add_new_dataframe(
        CENSUS_SUMMARY_NAME,
        schema=pa.Schema.from_pandas(df, preserve_index=False),
        index_column_names=["soma_joinid"],
        domain=[(df["soma_joinid"].min(), df["soma_joinid"].max())],
    ) as summary:
        summary.write(pa.Table.from_pandas(df, preserve_index=False))


def create_census_info_organisms(
    info_collection: soma.Collection, experiment_builders: Sequence[ExperimentBuilder]
) -> None:
    logger.info("Create census organisms dataframe")

    df = pd.DataFrame.from_records(
        [
            {
                "organism_ontology_term_id": eb.specification.organism_ontology_term_id,
                "organism_label": eb.specification.label,
                "organism": eb.specification.name,
            }
            for eb in experiment_builders
        ]
    )
    df["soma_joinid"] = range(len(df))
    with info_collection.add_new_dataframe(
        CENSUS_INFO_ORGANISMS_NAME,
        schema=pa.Schema.from_pandas(df, preserve_index=False),
        index_column_names=["soma_joinid"],
        domain=[(df["soma_joinid"].min(), df["soma_joinid"].max())],
    ) as summary:
        summary.write(pa.Table.from_pandas(df, preserve_index=False))
