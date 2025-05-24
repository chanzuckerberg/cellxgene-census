"""census_transcriptformer driver.

- Queries Census for cells matching specified criteria
- Batches the cells into groups of up to k
- For each batch, retrieves the Census data as an h5ad file
- Runs `transcriptformer inference` on the h5ad file
- Inference runs as a background process while preparing the next batch h5ad
"""

import argparse
import logging
import os
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
    parser.add_argument(
        "--obs-lo", type=int, default=0, help="obs soma_joinid range start (inclusive)"
    )
    parser.add_argument(
        "--obs-hi", type=int, default=sys.maxsize, help="obs soma_joinid range stop (exclusive)"
    )
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[census_transcriptformer] %(asctime)s - %(levelname)s - %(message)s",
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
            h5ad = f"{obs_coords[0]}_{obs_coords[-1]}.h5ad"
            get_census_h5ad(census, args.organism, obs_coords=obs_coords, h5ad_filename=h5ad)
            logging.info(f"Generated {h5ad} in {time.time() - start_time:.2f}s")
            if bgproc is not None:
                wait_start = time.time()
                bgproc.wait()
                wait_time = time.time() - wait_start
                wait_log = logging.info if wait_time >= 1.0 else logging.warning
                wait_log(
                    f"Waited {time.time() - wait_start:.2f}s for transcriptformer inference on {last_h5ad}"
                )
                os.remove(last_h5ad)
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
            # TODO: run put_embeddings.py at this point (in its own venv)
            last_h5ad = h5ad
        if bgproc is not None:
            bgproc.wait()
            os.remove(last_h5ad)
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
    """Yield obs soma_joinid's matching the given criteria, in batches of up to k."""
    obs_df = cellxgene_census.get_obs(
        census,
        organism,
        value_filter=obs_value_filter,
        coords=slice(obs_lo, obs_hi - 1),
        column_names=("soma_joinid",),
    )
    obs_ids = obs_df["soma_joinid"].astype(int).tolist()
    if obs_mod is not None:
        obs_ids = [id for id in obs_ids if id % obs_mod == 0]
    logging.info(
        f"Begin processing {len(obs_ids)} cells in batches of up to {k};"
        f" obs soma_joinid min={obs_lo} max={obs_hi}"
    )
    for i in range(0, len(obs_ids), k):
        yield obs_ids[i : i + k]


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


class CheckedPopen(subprocess.Popen):
    def wait(self, *args, **kwargs):
        rc = super().wait(*args, **kwargs)
        if rc != 0:
            raise subprocess.CalledProcessError(rc, self.args)
        return rc


if __name__ == "__main__":
    main()
