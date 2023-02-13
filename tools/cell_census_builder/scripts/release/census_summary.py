import argparse
import sys

import cell_census
import pandas as pd

from tools.cell_census_builder.globals import CENSUS_DATA_NAME, CENSUS_INFO_NAME


def display_summary(census_version: str) -> int:
    with cell_census.open_soma(census_version=census_version) as census:
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
        print()

        return 0


def display_diff(census_version: str, previous_census_version: str) -> int:
    with cell_census.open_soma(census_version=census_version) as census, cell_census.open_soma(
        census_version=previous_census_version
    ) as previous_census:
        prev_datasets = previous_census[CENSUS_INFO_NAME]["datasets"].read().concat().to_pandas()
        curr_datasets = census[CENSUS_INFO_NAME]["datasets"].read().concat().to_pandas()

        # Datasets removed, added
        curr_datasets_ids = set(curr_datasets["dataset_id"])
        prev_dataset_ids = set(prev_datasets["dataset_id"])

        added_datasets = curr_datasets_ids - prev_dataset_ids
        removed_datasets = prev_dataset_ids - curr_datasets_ids
        if added_datasets:
            print("Datasets that were added")
            added_datasets_df = curr_datasets[curr_datasets["dataset_id"].isin(added_datasets)]
            print(added_datasets_df[["dataset_id", "dataset_title", "collection_name"]])
        else:
            print("No datasets were added")
        print()

        if removed_datasets:
            print("Datasets that were removed")
            removed_datasets_df = prev_datasets[prev_datasets["dataset_id"].isin(removed_datasets)]
            print(removed_datasets_df[["dataset_id", "dataset_title", "collection_name"]])
        else:
            print("No datasets were removed")
        print()

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
            print()

        # Total cell count deltas by experiment (mouse, human)

        for organism in census[CENSUS_DATA_NAME]:
            curr_count = census[CENSUS_DATA_NAME][organism].obs.count
            prev_count = previous_census[CENSUS_DATA_NAME][organism].obs.count
            print(
                f"Previous {organism} cell count: {prev_count}, current {organism} cell count: {curr_count}, delta {curr_count - prev_count}"
            )
            print()

        # Deltas between summary_cell_counts dataframes
        y = census["census_info"]["summary_cell_counts"].read().concat().to_pandas()
        y = y.set_index(["organism", "category", "ontology_term_id"])

        z = previous_census["census_info"]["summary_cell_counts"].read().concat().to_pandas()
        z = z.set_index(["organism", "category", "ontology_term_id"])

        w = y.join(z, lsuffix="_prev", rsuffix="_curr")
        delta = w.loc[w["total_cell_count_prev"] != w["total_cell_count_curr"]][
            ["total_cell_count_prev", "total_cell_count_curr"]
        ].reset_index()
        if not delta.empty:
            print("Summary delta - total cell counts")
            print(delta)
            print()

        delta = w.loc[w["unique_cell_count_prev"] != w["unique_cell_count_curr"]][
            ["unique_cell_count_prev", "unique_cell_count_curr"]
        ].reset_index()
        if not delta.empty:
            print("Summary delta - unique cell counts")
            print(delta)
            print()

        # Genes removed, added
        for organism in census[CENSUS_DATA_NAME]:
            curr_genes = census[CENSUS_DATA_NAME][organism].ms["RNA"].var.read().concat().to_pandas()
            prev_genes = previous_census[CENSUS_DATA_NAME][organism].ms["RNA"].var.read().concat().to_pandas()

            new_genes = set(curr_genes["feature_id"]) - set(prev_genes["feature_id"])
            if new_genes:
                print("Genes added")
                print(new_genes)
            else:
                "No genes were added."
                print()

            removed_genes = set(prev_genes["feature_id"]) - set(curr_genes["feature_id"])
            if removed_genes:
                print("Genes removed")
                print(removed_genes)
            else:
                "No genes were removed."
                print()

        return 0


def create_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cell_census_summary")
    parser.add_argument("-c", "--census-version", default="latest", help="Version of the census. Defaults to latest")
    subparsers = parser.add_subparsers(required=True, dest="subcommand")

    # BUILD
    subparsers.add_parser("summarize", help="Summarize the cell census")

    # VALIDATE
    diff_parser = subparsers.add_parser("diff", help="Shows the diff with a previous census version")
    diff_parser.add_argument("-p", "--previous-version", help="Version of the census to diff")

    return parser


def main() -> int:
    parser = create_args_parser()
    args = parser.parse_args()
    assert args.subcommand in ["summarize", "diff"]

    if args.subcommand == "summarize":
        return display_summary(args.census_version)
    elif args.subcommand == "diff":
        return display_diff(args.census_version, args.previous_version)

    return 0


if __name__ == "__main__":
    sys.exit(main())
