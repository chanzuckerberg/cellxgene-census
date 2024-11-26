import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import tiledbsoma as soma

from .globals import CENSUS_SUMMARY_CELL_COUNTS_NAME, CENSUS_SUMMARY_CELL_COUNTS_TABLE_SPEC

logger = logging.getLogger(__name__)


def create_census_summary_cell_counts(
    info_collection: soma.Collection, per_experiment_summary: Sequence[pd.DataFrame]
) -> None:
    """Save per-category counts as the census_summary_cell_counts SOMA dataframe."""
    logger.info("Creating census_summary_cell_counts")
    df = (
        pd.concat(per_experiment_summary, ignore_index=True)
        .drop(columns=["dataset_id"])
        .groupby(by=["organism", "category", "ontology_term_id"], as_index=False, observed=True)
        .agg({"unique_cell_count": "sum", "total_cell_count": "sum", "label": "first"})
    )
    df["soma_joinid"] = df.index.astype(np.int64)
    df = CENSUS_SUMMARY_CELL_COUNTS_TABLE_SPEC.recategoricalize(df)

    schema = CENSUS_SUMMARY_CELL_COUNTS_TABLE_SPEC.to_arrow_schema(df)

    # write to a SOMA dataframe
    with info_collection.add_new_dataframe(
        CENSUS_SUMMARY_CELL_COUNTS_NAME,
        schema=schema,
        index_column_names=["soma_joinid"],
        domain=[(df["soma_joinid"].min(), df["soma_joinid"].max())],
    ) as cell_counts:
        cell_counts.write(pa.Table.from_pandas(df, preserve_index=False, schema=schema))


def init_summary_counts_accumulator() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "dataset_id": pd.Series([], dtype=str),
            **{
                field.name: pd.Series([], dtype=field.to_pandas_dtype(ignore_dict_type=True))  # type: ignore[arg-type]
                for field in CENSUS_SUMMARY_CELL_COUNTS_TABLE_SPEC.fields
            },
        }
    )


def accumulate_summary_counts(current: pd.DataFrame, obs_df: pd.DataFrame) -> pd.DataFrame:
    """Add summary counts to the census_summary_cell_counts dataframe."""
    assert "dataset_id" in obs_df

    if len(obs_df) == 0:
        return current

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

    dataset_id = obs_df.iloc[0].dataset_id

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
                observed=True,
                aggfunc="sum",  # noop: each element is unique. Necessary to prevent cast from int to float by default aggfunc (mean)
            )
        )
        if True not in counts:
            counts[True] = 0
        if False not in counts:
            counts[False] = 0

        counts["category"] = term_label if term_label is not None else term_id
        counts["unique_cell_count"] = counts[True]
        counts["total_cell_count"] = counts[True] + counts[False]
        counts["dataset_id"] = dataset_id
        counts = counts.drop(columns=[True, False]).reset_index()
        dfs.append(counts)

    all = pd.DataFrame(
        data={
            "dataset_id": [dataset_id],
            "organism": [obs_df.iloc[0].organism],
            "ontology_term_id": ["na"],
            "label": ["na"],
            "category": ["all"],
            "unique_cell_count": [dfs[0].unique_cell_count.sum()],
            "total_cell_count": [dfs[0].total_cell_count.sum()],
        }
    )

    return pd.concat([current, all, *dfs], ignore_index=True)
