"""driver for `transcriptformer inference`.

- Reads planner's JSON file with obs (cell) soma_joinid's to process.
- Splits the cells into "megabatches" (as distinct from torch batches)
- For each megabatch, retrieves the Census data as an h5ad file
- Runs `transcriptformer inference` on the h5ad file
- Calls put_embeddings.py to write the embeddings into a tiledbsoma.SparseNDArray
- Inference runs as a background process while we're writing the last megabatch of embeddings and
  preparing the next megabatch h5ad, so that we should have the next megabatch ready as soon as the
  previous one finishes.
"""

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import time

import cellxgene_census

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][inference_driver][%(levelname)s] - %(message)s",
)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Generate TranscriptFormer embeddings for Census")
    parser.add_argument("plan_json", type=str, help="shard JSON file from planner")
    parser.add_argument("array", type=str, help="existing tiledbsoma.SOMASparseNDArray for output")
    parser.add_argument(
        "--model",
        type=str,
        default="tf_metazoa",
        choices=["tf_metazoa", "tf_exemplar", "tf_sapiens"],
    )
    parser.add_argument("--megabatch-size", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    with open(args.plan_json) as f:
        plan = json.load(f)

    with cellxgene_census.open_soma(uri=plan["census_uri"]) as census:
        # plan megabatches
        megabatches = list(
            plan_megabatches(
                plan["obs_ids"],
                args.megabatch_size,
            )
        )
        bgproc = None
        last_h5ad = None
        for i, obs_coords in enumerate(megabatches):
            start_time = time.time()
            assert len(obs_coords) > 0
            # prepare current megabatch h5ad
            h5ad = f"{obs_coords[0]}_{obs_coords[-1]}.h5ad"
            get_census_h5ad(census, plan["organism"], obs_coords=obs_coords, h5ad_filename=h5ad)
            logging.info(
                f"Generated {h5ad} in {time.time() - start_time:.2f}s" f" (megabatch {i+1} of {len(megabatches)})"
            )
            if bgproc is not None:
                # wait for inference on the last megabatch to finish (we expect to be ready well
                # before bgproc)
                wait_start = time.time()
                bgproc.wait()
                wait_time = time.time() - wait_start
                wait_log = logging.info if wait_time >= 1.0 else logging.warning
                wait_log(f"Waited {time.time() - wait_start:.2f}s for transcriptformer inference on {last_h5ad}")
                os.remove(last_h5ad)
            # start `transcriptformer inference` on the current megabatch
            cmd = [
                "transcriptformer",
                "inference",
                "--checkpoint-path",
                os.path.join(HERE, "checkpoints", args.model),
                "--data-file",
                h5ad,
                "--output-path",
                f"./inference_{h5ad}",
                "--batch-size",
                str(args.batch_size),
            ]
            logging.info(f"Starting: {' '.join(cmd)}")
            bgproc = CheckedPopen(cmd)
            # with inference running, write the last megabatch embeddings to the output array
            if last_h5ad is not None:
                put_embeddings(f"./inference_{last_h5ad}/embeddings.h5ad", args.array)
            last_h5ad = h5ad
        if bgproc is not None:
            bgproc.wait()
            os.remove(last_h5ad)
            put_embeddings(f"./inference_{last_h5ad}/embeddings.h5ad", args.array)
        logging.info("DONE")


def plan_megabatches(obs_ids, megabatch_size):
    """Subdivide the plan list of cell IDs into "megabatches". This ensures
    `transcriptformer inference` runs in limited memory, and lets us pipeline the inference with
    reading the Census data and writing back the output embeddings."""
    n = len(obs_ids)
    num_mbatches = math.ceil(n / megabatch_size)
    base_size = n // num_mbatches
    rem = n % num_mbatches
    offset = 0
    for i in range(num_mbatches):
        size = base_size + (1 if i < rem else 0)
        yield obs_ids[offset : offset + size]
        offset += size
    assert offset == n


def get_census_h5ad(
    census,
    organism,
    *,
    obs_coords,
    h5ad_filename,
    obs_column_names=("soma_joinid", "assay"),
    var_column_names=("soma_joinid", "feature_id"),
):
    """Retrieve the Census raw RNA counts for the specified cell IDs (obs_coords) as a h5ad file
    suitable for `transcriptformer inference`."""
    adata = cellxgene_census.get_anndata(
        census,
        organism,
        obs_coords=obs_coords,
        obs_column_names=obs_column_names,
        var_column_names=var_column_names,
    )
    adata.var["ensembl_id"] = adata.var["feature_id"]
    adata.write_h5ad(h5ad_filename, compression="lzf")


def put_embeddings(embeddings_h5ad, array):
    """Run put_embeddings.py in a specially-prepared venv, to ensure the output array is readable
    by the Census embeddings curator tool."""
    subprocess.run(
        [
            os.path.join(HERE, "embeddings_tiledbsoma_venv", "bin", "python"),
            os.path.join(HERE, "put_embeddings.py"),
            embeddings_h5ad,
            array,
        ],
        check=True,
    )
    shutil.rmtree(os.path.dirname(embeddings_h5ad))


class CheckedPopen(subprocess.Popen):
    def wait(self, *args, **kwargs):
        rc = super().wait(*args, **kwargs)
        if rc != 0:
            raise subprocess.CalledProcessError(rc, self.args)
        return rc


if __name__ == "__main__":
    main()
