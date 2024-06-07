#!/usr/bin/env python3
# mypy: ignore-errors

import argparse
import logging
import multiprocessing
import tiledbsoma
import tiledb
import pyarrow as pa
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s [%(levelname)s] %(message)s")
logger = logging.getLogger(os.path.basename(__file__))
NPROC = multiprocessing.cpu_count()


def main(argv):
    args = parse_arguments(argv)

    if os.path.exists(args.output_file):
        logger.error("output directory already exists: " + args.output_file)
        raise FileExistsError()

    print((args.output_file))

    tiledbsoma.SparseNDArray.create(
        args.output_file,
        type=pa.float32(),
        shape=(2 ** 31 - 2, args.emb_dim),
        context=tiledbsoma.options.SOMATileDBContext(
            tiledb_ctx=tiledb.Ctx({"vfs.s3.region": args.s3_region})
        )
    ).close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Creates an empty tiledbsoma array with max row capacitiy and specified number of columns"
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        help="number of columns for array",
        required=True,
    )
    parser.add_argument(
        "--s3-region",
        type=str,
        default="us-west-2",
        help="AWS S3 region for tiledb context",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="output tiledbsoma array",
    )

    args = parser.parse_args(argv[1:])
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
