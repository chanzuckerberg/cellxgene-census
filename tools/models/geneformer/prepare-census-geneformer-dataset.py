#!/usr/bin/env python3
# mypy: ignore-errors

import argparse
import functools
import json
import logging
import math
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

# TODO: switch to https://github.com/chanzuckerberg/cellxgene-ontology-guide
from helpers.ontology_mapper import CellSubclassMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s [%(levelname)s] %(message)s")
logger = logging.getLogger(os.path.basename(__file__))


def main(argv):
    args = parse_arguments(argv)

    if os.path.exists(args.output_dir):
        logger.error("output directory already exists: " + args.output_dir)
        return 1

    # select cells
    if "://" in args.census_version:
        census_uri = args.census_version
    else:
        census_uri = cellxgene_census.get_census_version_description(args.census_version)["soma"]["uri"]
        logger.info(f"resolved census version {args.census_version} to {census_uri}")
    with cellxgene_census.open_soma(uri=census_uri) as census:
        obs_df = select_cells(
            census["census_data"]["homo_sapiens"], args.value_filter, args.percentage_data, args.sampling_column, args.N
        )

    logger.info(f"tokenizing {len(obs_df)} cells...")
    # build dataset (parallelizing across shards, if so configured)
    # NOTE: originally we made one big Dataset and later used its built-in shard() method, but we
    # found that didn't use disk I/O efficiently (reading a shard read the whole dataset) so
    # switched to sharding into separate datasets.
    tasks = [(obs_df, args.output_dir)]
    if args.shards > 1:
        obs_dfs = np.array_split(obs_df, args.shards)
        digits = math.ceil(math.log10(len(obs_dfs)))
        tasks = [
            (obs_dfs[i], os.path.join(args.output_dir, "shard-" + str(i).zfill(digits))) for i in range(len(obs_dfs))
        ]
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    with multiprocessing.Pool(
        processes=8, initializer=init_worker
    ) as pool:  # NOTE: keep processes= small due to memory usage
        pool.map(functools.partial(build_dataset, census_uri, args.obs_columns, args.tokenizer_kwargs), tasks)

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
        "--tokenizer-kwargs", type=json.loads, default={}, help="additional kwargs to pass to GeneformerTokenizer"
    )
    parser.add_argument("--shards", type=int, default=1, help="output dataset shards (default: 1)")
    parser.add_argument(
        "-v", "--census-version", type=str, default="stable", help='Census release or URI to query (default: "stable")'
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
    """Select the desired cells from the human census experiment.

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
    subclass_counts = None
    try:
        mapper = CellSubclassMapper(map_orphans_to_class=True)
        obs_df["cell_subclass_ontology_term_id"] = obs_df["cell_type_ontology_term_id"].map(
            # if CellSubclassMapper doesn't find a subclass, just use the cell type itself
            lambda it: (mapper.get_top_high_level_term(it) or it) if it != "unknown" else it
        )
        obs_df["cell_subclass"] = obs_df["cell_subclass_ontology_term_id"].map(
            lambda it: mapper.get_label_from_id(it) if it != "unknown" else it
        )
        subclass_counts = Counter(obs_df["cell_subclass"])
        logger.info(
            f"cell subclasses ({len(subclass_counts)}): {json.dumps(subclass_counts)}"
            + f" (compare to {len(obs_df['cell_type_ontology_term_id'].unique())} cell_types)"
        )
    except Exception:
        logger.exception("failed to annotate cell subclasses")

    # further downsample by sampling_column, if requested
    if N:
        sampling_counts = Counter(obs_df[sampling_column])
        if sampling_column != "cell_subclass":
            logger.info(f"initial counts of {sampling_column}: {json.dumps(sampling_counts)}")
        obs_df = obs_df.groupby(sampling_column).apply(lambda x: x.sample(min(len(x), N)))
        sampling_counts = Counter(obs_df[sampling_column])
        logger.info(f"after downsampling to at most {N} examples per {sampling_column}: {json.dumps(sampling_counts)}")
        if subclass_counts is not None:
            subclass_counts = Counter(obs_df["cell_subclass"])
            logger.info(f"downsampled cell subclasses ({len(subclass_counts)}): {json.dumps(subclass_counts)}")

    obs_df.set_index("soma_joinid", inplace=True)
    return obs_df


worker_soma_context = None


def init_worker():
    global worker_soma_context
    worker_soma_context = tiledbsoma.SOMATileDBContext()


def build_dataset(census_uri, obs_columns, tokenizer_kwargs, task):
    """Given obs_df from select_cells (or subset thereof), build the Geneformer dataset and save to output_dir."""
    obs_df = task[0]
    output_dir = task[1]

    # open human census
    with cellxgene_census.open_soma(uri=census_uri, context=worker_soma_context) as census:
        # use GeneformerTokenizer to build dataset of those cells
        with GeneformerTokenizer(
            census["census_data"]["homo_sapiens"],
            obs_query=tiledbsoma.AxisQuery(coords=(np.array(obs_df.index),)),
            obs_attributes=[
                # cell_subclass isn't yet in Census (select_cells() added it to obs_df for us), so
                # exclude from the experiment axis query
                it
                for it in obs_columns
                if it not in ("cell_subclass", "cell_subclass_ontology_term_id")
            ],
            **tokenizer_kwargs,
        ) as tokenizer:
            dataset = tokenizer.build()

    # add back cell_subclass from obs_df
    def add_cell_subclass(it):
        ans = {}
        if "cell_subclass_ontology_term_id" in obs_columns:
            ans["cell_subclass_ontology_term_id"] = obs_df.loc[it["soma_joinid"]]["cell_subclass_ontology_term_id"]
        if "cell_subclass" in obs_columns:
            ans["cell_subclass"] = obs_df.loc[it["soma_joinid"]]["cell_subclass"]
        return ans

    if "cell_subclass" in obs_df and "cell_subclass_ontology_term_id" in obs_df:
        dataset = dataset.map(add_cell_subclass)

    # save to output_dir
    dataset.save_to_disk(output_dir)
    logger.info("saved " + output_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
