#!/bin/python3

import yaml
import subprocess

def add_args(opts, args):
    for opt_key, opt_val in opts.items():
        if opt_key == "uri":
            args.append(opt_val)
        elif opt_key == "commands":
            continue
        elif isinstance(opt_val, bool):
            args.append(f"--{opt_key}")
        else:
            args.append(f"--{opt_key}")
            args.append(opt_val)


with open("build-census.yaml") as y:
    args = ["python3", "-m", "tools.cell_census_builder"]
    config = yaml.safe_load(y)
    builder = config["census-builder"]
    uri = builder["uri"]
    add_args(builder, args)
    commands = builder["commands"]
    for cmd, opts in commands.items():
        subcommand_args = args.copy()
        subcommand_args.append(cmd)
        add_args(opts, subcommand_args)
        print(subcommand_args)
        subprocess.call(subcommand_args)