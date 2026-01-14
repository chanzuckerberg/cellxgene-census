# mypy: ignore-errors
"""census_transcriptformer planning script.

1. Query Census for all obs soma_joinid's matching specified criteria.
2. Split them up into desired number of shards.
3. Write each shard into a plan_*.json file, to be processed by an inference worker.
4. Write the full list of obs soma_joinid's to a text file.
"""

import argparse
import json
import logging

import cellxgene_census

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][planner][%(levelname)s] - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Generate TranscriptFormer embeddings for Census")
    parser.add_argument(
        "--census-uri",
        type=str,
        default=("s3://cellxgene-census-public-us-west-2/cell-census/2025-01-30/soma/"),
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="Homo sapiens",
        choices=("Homo sapiens", "Mus musculus"),
    )
    parser.add_argument("--value-filter", type=str)
    parser.add_argument("--mod", type=int)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    with cellxgene_census.open_soma(uri=args.census_uri) as census:
        obs_df = cellxgene_census.get_obs(
            census,
            args.organism,
            value_filter=args.value_filter,
            column_names=("soma_joinid",),
        )
    obs_ids = obs_df["soma_joinid"].astype(int).tolist()
    if args.mod is not None:
        obs_ids = [id for id in obs_ids if id % args.mod == 0]
    n = len(obs_ids)
    if n == 0:
        logging.warning("No observation IDs found matching criteria. Exiting.")
        return
    if args.shards > n:
        logging.warning("Reducing shards to %d", n)
        args.shards = n
    logging.info("Splitting %d obs ids into %d shards", n, args.shards)
    base_size = n // args.shards
    rem = n % args.shards
    offset = 0
    obs_id_shards = []
    for i in range(args.shards):
        size = base_size + (1 if i < rem else 0)
        obs_id_shards.append(obs_ids[offset : offset + size])
        offset += size
    assert offset == n

    chk = 0
    with open("obs_ids.txt", "w") as obs_ids_txt:
        for obs_ids in obs_id_shards:
            filename = f"plan_{obs_ids[0]}_{obs_ids[-1]}.json"
            logging.info("Writing %d obs ids to %s", len(obs_ids), filename)
            with open(filename, "w") as f:
                obj = {
                    "census_uri": args.census_uri,
                    "organism": args.organism,
                    "value_filter": args.value_filter,
                    "mod": args.mod,
                    "obs_ids": obs_ids,
                }
                json.dump(obj, f)
            for obs_id in obs_ids:
                obs_ids_txt.write(f"{obs_id}\n")
                chk += 1
    assert chk == n


if __name__ == "__main__":
    main()
