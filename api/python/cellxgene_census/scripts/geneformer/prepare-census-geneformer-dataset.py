#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import numpy as np
import tiledbsoma

import cellxgene_census
from cellxgene_census.experimental.ml.huggingface import GeneformerTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s [%(levelname)s] %(message)s")
logger = logging.getLogger(os.path.basename(__file__))


def main(argv):
    args = parse_arguments(argv)

    if os.path.exists(args.output_dir):
        logger.error("output directory already exists: " + args.output_dir)
        return 1

    # open human census
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        census_human = census["census_data"]["homo_sapiens"]

        # select the cell id's to include
        coords = select_cells(census_human, args.value_filter, args.percentage_data)

        # use GeneformerTokenizer to build dataset of those cells
        with GeneformerTokenizer(
            census_human, obs_query=tiledbsoma.AxisQuery(coords=(coords,)), obs_attributes=args.obs_columns
        ) as tokenizer:
            logger.info("tokenizing...")
            dataset = tokenizer.build()

        # write them to output_dir (note: the Dataset tools will have spooled to disk already, so
        # this should just be copying it to the desired location)
        logger.info("writing Dataset to " + args.output_dir)
        dataset.save_to_disk(args.output_dir)

    logger.info("done")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Prepare Geneformer Dataset from CELLxGENE Census")
    parser.add_argument(
        "-f",
        "--value-filter",
        type=str,
        default="is_primary_data==True",
        help="cell filter (default: is_primary_data==True)",
    )
    parser.add_argument(
        "-p", "--percentage-data", type=int, default=10, help="percent of human primary cells (default: 10)"
    )
    parser.add_argument(
        "-c",
        "--obs-columns",
        type=str,
        default="cell_type",
        help="cell attributes to include in dataset (comma-separated; default: cell_type)",
    )
    parser.add_argument(
        "-v", "--census-version", type=str, default="latest", help='Census release to query (default: "latest")'
    )
    parser.add_argument("output_dir", type=str, help="output directory (must not already exist)")

    args = parser.parse_args(argv)

    if args.obs_columns:
        args.obs_columns = args.obs_columns.split(",")

    logger.info("arguments: " + str(vars(args)))
    return args


def select_cells(census_human: tiledbsoma.Experiment, value_filter: str, percentage_data: int) -> np.ndarray:
    assert 1 <= percentage_data and percentage_data <= 100
    obs_df = census_human["obs"].read(value_filter=value_filter, column_names=["soma_joinid"]).concat().to_pandas()
    logger.info(f"total cells matching value_filter: {format(len(obs_df), ',')}")
    if percentage_data < 100:
        obs_df = obs_df.sample(n=int(len(obs_df) * (percentage_data / 100.0)))
    logger.info(f"sampled cells: {format(len(obs_df), ',')}")
    return obs_df["soma_joinid"].values


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
