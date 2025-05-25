"""census_transcriptformer driver.

- Queries Census for cells matching specified criteria
- Groups the cells in "megabatches" of up to k (megabatch in contrast to the torch batch size)
- For each megabatch, retrieves the Census data as an h5ad file
- Runs `transcriptformer inference` on the h5ad file
- Calls put_embeddings.py to write the embeddings into a tiledbsoma.SparseNDArray
- Inference runs as a background process while we're writing the last megabatch of embeddings and
  preparing the next megabatch h5ad
"""

import argparse
import logging
import math
import os
import shutil
import subprocess
import sys
import time

import cellxgene_census

HERE = os.path.dirname(os.path.abspath(__file__))


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
    parser.add_argument("--obs-lo", type=int, default=0, help="obs soma_joinid range start (inclusive)")
    parser.add_argument("--obs-hi", type=int, default=sys.maxsize, help="obs soma_joinid range stop (exclusive)")
    parser.add_argument("--obs-value-filter", type=str)
    parser.add_argument("--obs-mod", type=int)
    parser.add_argument("-k", type=int, default=25000)
    parser.add_argument(
        "--model",
        type=str,
        default="metazoa",
        choices=["metazoa", "exemplar", "tf_sapiens"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--array", type=str, required=True, help="existing tiledbsoma.SOMASparseNDArray for output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][census_transcriptformer][%(levelname)s] - %(message)s",
    )
    with cellxgene_census.open_soma(uri=args.census_uri) as census:
        bgproc = None
        last_h5ad = None
        for obs_coords in plan_obs_coords(
            census,
            args.organism,
            obs_lo=args.obs_lo,
            obs_hi=args.obs_hi,
            obs_value_filter=args.obs_value_filter,
            obs_mod=args.obs_mod,
            k=args.k,
        ):
            start_time = time.time()
            assert len(obs_coords) > 0
            # prepare current megabatch h5ad
            h5ad = f"{obs_coords[0]}_{obs_coords[-1]}.h5ad"
            get_census_h5ad(census, args.organism, obs_coords=obs_coords, h5ad_filename=h5ad)
            logging.info(f"Generated {h5ad} in {time.time() - start_time:.2f}s")
            if bgproc is not None:
                # wait for inference on the last megabatch to finish
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
            put_embeddings(f"./inference_{last_h5ad}/embeddings.h5ad", args.array)
            last_h5ad = h5ad
        if bgproc is not None:
            bgproc.wait()
            os.remove(last_h5ad)
            put_embeddings(f"./inference_{last_h5ad}/embeddings.h5ad", args.array)
        logging.info("DONE")


def plan_obs_coords(
    census,
    organism,
    *,
    obs_lo,
    obs_hi,
    obs_value_filter,
    obs_mod,
    k,
):
    """Yield obs soma_joinid's matching the given criteria, in "megabatches" of up to k."""
    obs_df = cellxgene_census.get_obs(
        census,
        organism,
        value_filter=obs_value_filter,
        coords=slice(obs_lo, obs_hi - 1),  # NB: TileDB expects closed intervals
        column_names=("soma_joinid",),
    )
    obs_ids = obs_df["soma_joinid"].astype(int).tolist()
    if obs_mod is not None:
        obs_ids = [id for id in obs_ids if id % obs_mod == 0]
    n = len(obs_ids)
    logging.info(
        f"Begin processing {len(obs_ids)} cells in megabatches of up to {k}; obs soma_joinid min={obs_lo} max={obs_hi}"
    )
    if n == 0:
        return
    num_mbatches = math.ceil(n / k)
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
