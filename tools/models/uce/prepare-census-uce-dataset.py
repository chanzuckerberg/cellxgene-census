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
import os
import numpy as np
import tiledbsoma

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s [%(levelname)s] %(message)s")
logger = logging.getLogger(os.path.basename(__file__))
NPROC = multiprocessing.cpu_count()


def main(argv):
    args = parse_arguments(argv)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        logger.warning("output directory already exists: " + args.output_dir)
        #logger.error("output directory already exists: " + args.output_dir)
        #return 1


    # open human census
    with cellxgene_census.open_soma(census_version=args.census_version) as census:

        # select the cell id's to include
        coords = get_soma_joinid_slice(census["census_data"]["homo_sapiens"], args.part, args.parts)

        adata = cellxgene_census.get_anndata(
            census,
            "homo_sapiens",
            "RNA",
            obs_coords=coords,
            column_names={"obs": args.obs_columns},
        )

        adata.var_names = adata.var["feature_name"]
        adata.write_h5ad(os.path.join(args.output_dir, f"anndata_uce_{args.part}.h5ad"))

    logger.info(subprocess.run(["du", "-sh", args.output_dir], stdout=subprocess.PIPE).stdout.decode().strip())


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Prepare Geneformer Dataset from CELLxGENE Census")
    parser.add_argument(
        "-c",
        "--obs-columns",
        type=str,
        default="cell_type",
        help="cell attributes to include in anndata for UCE (comma-separated)",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="Zero-based Census partition to execute, effectively the nth H5AD to created."
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=1000,
        help="Number of total data partitions for Census, effectively number of H5AD files that will be created."
    )
    parser.add_argument(
        "-v", "--census-version", type=str, default="latest", help='Census release to query (default: "latest")'
    )
    parser.add_argument("output_dir", type=str, help="output directory (must not already exist)")

    args = parser.parse_args(argv[1:])

    if not (args.part >= 0 and args.parts is not None and args.parts > args.part):
        parser.error("--part must be nonnegative and less than --parts")

    if args.obs_columns:
        args.obs_columns = [s.strip() for s in args.obs_columns.split(",")]
    else:
        args.obs_columns = []

    if "soma_joinid" not in args.obs_columns:
        args.obs_columns.append("soma_joinid")

    logger.info("arguments: " + str(vars(args)))
    return args


def get_soma_joinid_slice(soma_experiment: tiledbsoma.Experiment, part, parts):
    """"Gets list of contiguous soma joinids for the corresponding part in the context of the total numboer of parts.
    Assumes soma joinids are an incremental list of integers starting a 0.
    """

    n_obs = len(soma_experiment.obs)
    part_size = int(n_obs/parts)
    start = part * part_size
    end = start + part_size - 1

    if part == parts:
        end = n_obs - 1 # tiledbsoma slices are inclusive in both ends

    return slice(start, end)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
