#!/usr/bin/env python3
# mypy: ignore-errors

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))  # to find ./helpers

import cellxgene_census
import numpy as np
import tiledbsoma
from cellxgene_census.experimental.ml.huggingface import GeneformerTokenizer
from helpers.ontology_mapper import CellSubclassMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s [%(levelname)s] %(message)s")
logger = logging.getLogger(os.path.basename(__file__))
NPROC = multiprocessing.cpu_count()


def main(argv):
    args = parse_arguments(argv)

    if os.path.exists(args.output_dir):
        logger.error("output directory already exists: " + args.output_dir)
        return 1

    # open human census
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        census_human = census["census_data"]["homo_sapiens"]

        # select the cell id's to include
        obs_df = select_cells(census_human, args.value_filter, args.percentage_data, args.sampling_column, args.N)
        coords = np.array(obs_df.index)

        # use GeneformerTokenizer to build dataset of those cells
        with GeneformerTokenizer(
            census_human,
            obs_query=tiledbsoma.AxisQuery(coords=(coords,)),
            obs_attributes=list(
                # cell_subclass isn't yet in Census (select_cells() added it to obs_df for us), so
                # exclude from the experiment axis query
                it
                for it in args.obs_columns
                if it not in ("cell_subclass", "cell_subclass_ontology_term_id")
            ),
        ) as tokenizer:
            logger.info(f"tokenizing {len(coords)} cells...")
            dataset = tokenizer.build()

    # add back cell_subclass
    if "cell_subclass_ontology_term_id" in args.obs_columns:
        dataset = dataset.map(
            lambda it: {
                "cell_subclass_ontology_term_id": obs_df.loc[it["soma_joinid"]]["cell_subclass_ontology_term_id"]
            },
            num_proc=NPROC,
        )
    if "cell_subclass" in args.obs_columns:
        dataset = dataset.map(
            lambda it: {"cell_subclass": obs_df.loc[it["soma_joinid"]]["cell_subclass"]}, num_proc=NPROC
        )
    logger.info(str(dataset))
    if len(dataset):
        logger.info(dataset[0])

    # write them to output_dir (note: the Dataset tools will have spooled to disk already, so
    # this should just be copying it to the desired location)
    logger.info("writing Dataset to " + args.output_dir)
    dataset.save_to_disk(args.output_dir)

    logger.info(subprocess.run(["du", "-sh", args.output_dir], stdout=subprocess.PIPE).stdout.decode().strip())


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
        "-p",
        "--percentage-data",
        type=int,
        default=100,
        help="percent of cells matching value_filter to include (default: 100)",
    )
    parser.add_argument(
        "-c",
        "--obs-columns",
        type=str,
        default="cell_type,cell_type_ontology_term_id,cell_subclass,cell_subclass_ontology_term_id",
        help="cell attributes to include in dataset (comma-separated)",
    )
    parser.add_argument(
        "--sampling-column",
        type=str,
        default="cell_subclass",
        help="column name to use for downsampling (default: cell_subclass)",
    )
    parser.add_argument(
        "-N", type=int, help="further downsample to no more than N examples per distinct value of sampling_column"
    )
    parser.add_argument(
        "-v", "--census-version", type=str, default="latest", help='Census release to query (default: "latest")'
    )
    parser.add_argument("output_dir", type=str, help="output directory (must not already exist)")

    args = parser.parse_args(argv[1:])

    if args.obs_columns:
        args.obs_columns = [s.strip() for s in args.obs_columns.split(",")]
    else:
        args.obs_columns = []

    if "soma_joinid" not in args.obs_columns:
        args.obs_columns.append("soma_joinid")
    if args.sampling_column not in args.obs_columns:
        args.obs_columns.append(args.sampling_column)

    logger.info("arguments: " + str(vars(args)))
    return args


def select_cells(census_human, value_filter, percentage_data, sampling_column, N):
    """
    Select the desired cells from the human census experiment.

    Return a pd.DataFrame indexed by soma_joinid with additional cell_subclass and cell_subclass_ontology_term_id
    attributes. These aren't currently provided in obs, so we derive them on the fly.
    """
    assert 1 <= percentage_data and percentage_data <= 100
    cols = ["soma_joinid", "cell_type_ontology_term_id"]
    if sampling_column not in ("cell_subclass", "cell_subclass_ontology_term_id") and sampling_column not in cols:
        cols.append(sampling_column)
    obs_df = census_human["obs"].read(value_filter=value_filter, column_names=cols).concat().to_pandas()
    logger.info(f"total cells matching value_filter: {format(len(obs_df), ',')}")
    if percentage_data < 100:
        obs_df = obs_df.sample(n=int(len(obs_df) * (percentage_data / 100.0)))
        logger.info(f"sampled cells: {format(len(obs_df), ',')}")

    # annotate cell subclasses
    logger.info("annotating cell subclasses...")
    mapper = CellSubclassMapper(map_orphans_to_class=True)
    obs_df["cell_subclass_ontology_term_id"] = obs_df["cell_type_ontology_term_id"].map(
        # if CellSubclassMapper doesn't find a subclass, just use the cell type itself
        lambda it: mapper.get_top_high_level_term(it)
        or it
    )
    obs_df["cell_subclass"] = obs_df["cell_subclass_ontology_term_id"].map(lambda it: mapper.get_label_from_id(it))
    subclass_counts = Counter(obs_df["cell_subclass"])
    logger.info(
        f"cell subclasses ({len(subclass_counts)}): {json.dumps(subclass_counts)}"
        + f" (compare to {len(obs_df['cell_type_ontology_term_id'].unique())} cell_types)"
    )

    # further downsample by sampling_column, if requested
    if N:
        sampling_counts = Counter(obs_df[sampling_column])
        if sampling_column != "cell_subclass":
            logger.info(f"initial counts of {sampling_column}: {json.dumps(sampling_counts)}")
        obs_df = obs_df.groupby(sampling_column).apply(lambda x: x.sample(min(len(x), N)))
        sampling_counts = Counter(obs_df[sampling_column])
        logger.info(f"after downsampling to at most {N} examples per {sampling_column}: {json.dumps(sampling_counts)}")
        subclass_counts = Counter(obs_df["cell_subclass"])
        logger.info(f"downsampled cell subclasses ({len(subclass_counts)}): {json.dumps(subclass_counts)}")

    obs_df.set_index("soma_joinid", inplace=True)
    return obs_df


if __name__ == "__main__":
    sys.exit(main(sys.argv))
