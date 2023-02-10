import sys

import cell_census
import pandas as pd

from tools.cell_census_builder.globals import CENSUS_DATA_NAME, CENSUS_INFO_NAME

if __name__ == "__main__":
    census_version = sys.argv[1] if len(sys.argv) > 1 else "latest"
    previous_census_version = sys.argv[2] if len(sys.argv) > 2 else None

    census = cell_census.open_soma(census_version=census_version)

    COLS_TO_QUERY = [
        ("soma_joinid", "cells"),
        ("dataset_id", "datasets"),
        ("cell_type_ontology_term_id", "cell types"),
        ("tissue_ontology_term_id", "tissues"),
        ("assay_ontology_term_id", "assays"),
    ]

    obs_df = {
        name: experiment.obs.read(column_names=[c[0] for c in COLS_TO_QUERY]).concat().to_pandas()
        for name, experiment in census[CENSUS_DATA_NAME].items()
    }

    # Use Pandas to summarize and display
    stats = [(organism, col[1], df[col[0]].nunique()) for organism, df in obs_df.items() for col in COLS_TO_QUERY]
    print(census["census_info"]["summary"].read().concat().to_pandas()[["label", "value"]].to_string(index=False))
    stats_df = pd.DataFrame(stats, columns=["organism", "attribute", "unique count"])
    display_stats_df = pd.pivot(stats_df, index=["organism"], columns=["attribute"], values=["unique count"])
    print(display_stats_df)

    # Calculate the diff with respect to the previous version (if specified)

    if previous_census_version is not None:
        previous_census = cell_census.open_soma(census_version=previous_census_version)

        prev_datasets = previous_census[CENSUS_INFO_NAME]["datasets"].read().concat().to_pandas()
        curr_datasets = census[CENSUS_INFO_NAME]["datasets"].read().concat().to_pandas()

        # Datasets removed, added
        curr_datasets_ids = set(curr_datasets["dataset_id"])
        prev_dataset_ids = set(prev_datasets["dataset_id"])

        added_datasets = curr_datasets_ids - prev_dataset_ids
        removed_datasets = prev_dataset_ids - curr_datasets_ids
        if added_datasets:
            print("Datasets that were added")
            for d in added_datasets:
                print(d)

        if removed_datasets:
            print("Datasets that were removed")
            for d in removed_datasets:
                print(d)

        # Datasets in both versions but that have differing cell counts
        joined = prev_datasets.join(
            curr_datasets.set_index("dataset_id"), on="dataset_id", lsuffix="_prev", rsuffix="_curr"
        )
        datasets_with_different_cell_counts = joined.loc[
            joined["dataset_total_cell_count_prev"] != joined["dataset_total_cell_count_curr"]
        ][["dataset_id", "dataset_total_cell_count_prev", "dataset_total_cell_count_curr"]]

        if not datasets_with_different_cell_counts.empty:
            print("Datasets that have a different cell count")
            print(datasets_with_different_cell_counts)

        # Total cell count deltas by experiment (mouse, human)
        for organism in ["homo_sapiens", "mus_musculus"]:
            curr_count = census[CENSUS_DATA_NAME][organism]["obs"].count
            prev_count = previous_census[CENSUS_DATA_NAME][organism]["obs"].count
            print(
                f"Previous {organism} cell count: {prev_count}, current {organism} cell count: {curr_count}, delta {curr_count - prev_count}"
            )

        # Deltas between summary_cell_counts dataframes

        # Genes removed, added
        for organism in ["homo_sapiens", "mus_musculus"]:
            curr_genes = census[CENSUS_DATA_NAME][organism]["ms"]["RNA"]["var"].read().concat().to_pandas()
            prev_genes = previous_census[CENSUS_DATA_NAME][organism]["ms"]["RNA"]["var"].read().concat().to_pandas()

            new_genes = set(curr_genes["feature_id"]) - set(prev_genes["feature_id"])
            if new_genes:
                print("Genes added")
                new_genes

            removed_genes = set(prev_genes["feature_id"]) - set(curr_genes["feature_id"])
            if removed_genes:
                print("Genes removed")
                removed_genes
