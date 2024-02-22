import argparse
import json
import logging
import re
from typing import Any

import attr
import profiler


def format_string(text: str) -> Any:
    return re.sub("\n", " ", text)


# The script takes a command and a database path and looks
# the performance anomalies in the performance history of that
# command across the profiled runs.

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("command", type=str)
parser.add_argument("db_path", type=str)

args = parser.parse_args()

# Processes the set of previously written logs
# The threshold (ratio) of allowable performance degradation between profiling runs
threshold = 1.10

db = profiler.data.S3ProfileDB(args.db_path)
command_profiles = db.find(f"{args.command}")


if len(command_profiles) >= 2:
    first_profile = command_profiles[0]
    curr_profile = command_profiles[-1]
    first_time = first_profile.elapsed_time_sec
    curr_time = curr_profile.elapsed_time_sec

    formatted_first_profile = json.dumps(format_string(str(attr.asdict(first_profile))))
    formatted_curr_profile = json.dumps(format_string(str(attr.asdict(curr_profile))))

    logging.info("****************************")
    logging.info(f"Current time {curr_time} vs first time {first_time}")
    logging.info("****************************")
    logging.info(f"First profile: {formatted_first_profile}")
    logging.info("****************************")
    logging.info(f"Current profile: {formatted_curr_profile}")
    logging.info("****************************")
    logging.info(
        f"TileDB version ver = first: {first_profile.tiledbsoma_version} curr: {curr_profile.tiledbsoma_version}"
    )
    if float(curr_time) > threshold * float(first_time):
        raise SystemExit(f"Major performance degradation detected on {args.benchmark}")

    if threshold * float(curr_time) < float(first_time):
        logging.info(f"Major performance increase detected on {args.command}")
