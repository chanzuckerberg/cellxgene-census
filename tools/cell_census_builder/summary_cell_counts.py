import logging
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import tiledbsoma as soma

from .globals import CENSUS_SUMMARY_CELL_COUNTS_COLUMNS, CENSUS_SUMMARY_CELL_COUNTS_NAME, SOMA_TileDB_Context
from .util import (
    anndata_ordered_bool_issue_853_workaround,
    pandas_dataframe_strings_to_ascii_issue_247_workaround,
    uricat,
)


def create_census_summary_cell_counts(
    info_collection: soma.Collection, per_experiment_summary: Sequence[pd.DataFrame]
) -> None:
    """
    Save per-category counts as the census_summary_cell_counts SOMA dataframe
    """
    logging.info("Creating census_summary_cell_counts")
    df = (
        pd.concat(per_experiment_summary, ignore_index=True)
        .drop(columns=["dataset_id"])
        .groupby(by=["organism", "category", "ontology_term_id"], as_index=False, observed=True)
        .agg({"unique_cell_count": "sum", "total_cell_count": "sum", "label": "first"})
    )
    df["soma_joinid"] = df.index.astype(np.int64)

    # TODO: work-around for TileDB-SOMA#274.  Remove when fixed.
    df = pandas_dataframe_strings_to_ascii_issue_247_workaround(df)
    df = anndata_ordered_bool_issue_853_workaround(df)

    # write to a SOMA dataframe
    summary_counts_uri = uricat(info_collection.uri, CENSUS_SUMMARY_CELL_COUNTS_NAME)
    summary_counts = soma.DataFrame(summary_counts_uri, context=SOMA_TileDB_Context())
    summary_counts.create(pa.Schema.from_pandas(df, preserve_index=False), index_column_names=["soma_joinid"])
    for batch in pa.Table.from_pandas(df, preserve_index=False).to_batches():
        summary_counts.write(batch)
    info_collection.set(CENSUS_SUMMARY_CELL_COUNTS_NAME, summary_counts, relative=True)


def init_summary_counts_accumulator() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "dataset_id": pd.Series([], dtype=str),
            **{
                name: pd.Series([], dtype=arrow_type.to_pandas_dtype())
                for name, arrow_type in CENSUS_SUMMARY_CELL_COUNTS_COLUMNS.items()
            },
        }
    )


def accumulate_summary_counts(current: pd.DataFrame, obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary counts to the census_summary_cell_counts dataframe
    """
    assert "dataset_id" in obs_df
    assert len(obs_df) > 0

    CATEGORIES = [
        # term_id, label
        ("cell_type_ontology_term_id", "cell_type"),
        ("assay_ontology_term_id", "assay"),
        ("tissue_ontology_term_id", "tissue"),
        ("disease_ontology_term_id", "disease"),
        ("self_reported_ethnicity_ontology_term_id", "self_reported_ethnicity"),
        ("sex_ontology_term_id", "sex"),
        ("tissue_general_ontology_term_id", "tissue_general"),
        (None, "suspension_type"),
    ]

    dfs = []
    for term_id, term_label in CATEGORIES:
        cats = []
        columns = {}
        assert term_id is not None or term_label is not None
        if term_id is not None:
            cats.append(term_id)
            columns.update({term_id: "ontology_term_id"})
        if term_label is not None:
            cats.append(term_label)
            columns.update({term_label: "label"})
        assert len(cats) > 0 and len(columns) > 0  # i.e., one or both of term or label are specified

        df = obs_df[["dataset_id", "organism", *cats, "is_primary_data"]].rename(columns=columns)
        if "label" not in df:
            df["label"] = "na"
        if "ontology_term_id" not in df:
            df["ontology_term_id"] = "na"

        counts = (
            df.value_counts()
            .to_frame(name="count")
            .reset_index(level="is_primary_data")
            .pivot_table(
                values="count",
                columns="is_primary_data",
                index=["organism", "ontology_term_id", "label"],
                fill_value=0,
            )
        )
        if True not in counts:
            counts[True] = 0
        if False not in counts:
            counts[False] = 0

        counts["category"] = term_label if term_label is not None else term_id
        counts["unique_cell_count"] = counts[True]
        counts["total_cell_count"] = counts[True] + counts[False]
        counts = counts.drop(columns=[True, False]).reset_index()
        dfs.append(counts)

    all = pd.DataFrame(
        data={
            "dataset_id": [obs_df.iloc[0].dataset_id],
            "organism": [obs_df.iloc[0].organism],
            "ontology_term_id": ["na"],
            "label": ["na"],
            "category": ["all"],
            "unique_cell_count": [dfs[0].unique_cell_count.sum()],
            "total_cell_count": [dfs[0].total_cell_count.sum()],
        }
    )
    return pd.concat([current, all, *dfs], ignore_index=True)
