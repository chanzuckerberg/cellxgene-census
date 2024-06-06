#!/usr/bin/env python3
# mypy: ignore-errors

import os
import sys
import shutil
import logging
import tempfile
import argparse
import subprocess


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s [%(levelname)s] %(message)s")
logger = logging.getLogger(os.path.basename(__file__))

def main(argv):
    args = parse_arguments(argv)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tiledbsoma_context = None
    if args.tiledbsoma:
        # prep tiledbsoma (and fail fast if there's a problem)
        import boto3
        import tiledb
        import tiledbsoma

        logger.info(f"loaded tiledbsoma=={tiledbsoma.__version__} tiledb=={tiledb.__version__}")

        aws_region = "us-west-2"
        try:
            aws_region = boto3.Session().region_name
        except Exception:  # noqa: BLE001
            pass
        tiledbsoma_context = tiledbsoma.options.SOMATileDBContext(
            tiledb_ctx=tiledb.Ctx(
                {
                    "py.init_buffer_bytes": 4 * 1024**3,
                    "soma.init_buffer_bytes": 4 * 1024**3,
                    "vfs.s3.region": aws_region,
                }
            )
        )

        with tiledbsoma.SparseNDArray.open(args.outfile, "r", context=tiledbsoma_context):
            # TODO: verify schema compatibility
            pass

    model_path = uce_33l_model_file("./", "./UCE/")
    uce_dir="./UCE/"

    with tempfile.TemporaryDirectory() as scratch_dir:
        # prepare the dataset, taking only one shard of it if so instructed
        dataset_path = prepare_dataset(args.dataset_dir, args.part)
        dataset_filename = os.path.basename(dataset_path)

        if not os.path.exists(os.path.join(uce_dir, dataset_filename)):
            os.makedirs(os.path.join(uce_dir, dataset_filename))

        dataset_path_uce = uce(
            dataset_path,
            uce_dir=uce_dir,
            relative_work_dir=dataset_filename,
            uce_33l_model_file=model_path,
        )

        shutil.move(os.path.join(uce_dir, dataset_path_uce), os.path.join(args.output_dir, dataset_filename))

        logger.info("Extracting embeddings...")

        # TODO generate embs

        if False:
            if args.tiledbsoma:
                import numpy as np
                import pyarrow as pa
                from scipy.sparse import coo_matrix

                # NOTE: embs_df has columns -named- 0, 1, ..., 511 as well as the requested features.
                embedding_dim = embs_df.shape[1] - len(args.features)
                logger.info(f"writing to tiledbsoma.SparseNDArray at {args.outfile}, embedding_dim={embedding_dim}...")
                with tiledbsoma.SparseNDArray.open(args.outfile, "w", context=tiledbsoma_context) as array:
                    dim0 = np.repeat(embs_df["soma_joinid"].values, embedding_dim)
                    dim1 = np.tile(np.arange(embedding_dim), len(embs_df))
                    data = embs_df.loc[:, range(embedding_dim)].values.flatten()
                    array.write(pa.SparseCOOTensor.from_scipy(coo_matrix((data, (dim0, dim1)))))
            else:
                logger.info(f"writing {args.outfile}...")
                # reorder embs_df columns and write to TSV
                cols = embs_df.columns.tolist()
                emb_cols = [col for col in cols if isinstance(col, int)]
                anno_cols = [col for col in cols if not isinstance(col, int)]
                embs_df = embs_df[anno_cols + emb_cols].set_index("soma_joinid").sort_index()
                embs_df.to_csv(args.outfile, sep="\t", header=True, index=True, index_label="soma_joinid")

        logger.info("SUCCESS")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Generate UCE embeddings for given cells dataset")
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="process only one shard of the data (zero-based index)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="directory with saved anndatas with format 'anndata_uce_{part}.h5ad'"
    )
    parser.add_argument(
        "--tiledbsoma",
        action="store_true",
        help="outfile is URI to an existing tiledbsoma.SparseNDArray to write into (instead of TSV file)",
    )
    parser.add_argument("output_dir", type=str, help="output directory (must not already exist)")


    args = parser.parse_args(argv[1:])

    logger.info("arguments: " + str(vars(args)))
    return args


def prepare_dataset(dataset_dir, part):
    dataset_path = os.path.join(dataset_dir, f"anndata_uce_{part}.h5ad")
    if not os.path.isfile(dataset_path):
        raise FileExistsError(f"{dataset_path} file does not exist")
    return dataset_path


def uce(h5ad, uce_dir, relative_work_dir, uce_33l_model_file, args=None):
    """
    Author: Mike Lin
    Run UCE eval_single_anndata.py.
    - it auto-fetches its dependent data files if needed (except for the 33-layer model file)
    - cwd needs to be the UCE repo for its relative default paths to work
    - leaves behind various intermediate files in work_dir
    - the intermediate files are extensions of the input h5ad basename
    - it reuses the intermediate files if they're already present (but doesn't check if they're
      newer than the input h5ad)
    """
    args = args or []
    name = os.path.splitext(os.path.basename(h5ad))[0]
    h5ad = os.path.join("../", h5ad)
    subprocess.run(
        [
            "python3",
            "eval_single_anndata.py",
            "--adata_path",
            h5ad,
            "--dir",
            (relative_work_dir + "/" if not relative_work_dir.endswith("/") else relative_work_dir),
            "--nlayers",
            "33",
            "--model_loc",
            uce_33l_model_file,
            *args,
        ],
        cwd=uce_dir,
        check=True,
    )
    # read output h5ad
    out_file = os.path.join(relative_work_dir, f"{name}_uce_adata.h5ad")
    return out_file


def uce_33l_model_file(pkg_tmpdir, uce_dir):
    """
    Author: Mike Lin
    Fetch the UCE 33-layer model file if needed.
    """
    relative_path = "model_files/33l_8ep_1024t_1280.torch"
    fn = os.path.join(uce_dir, relative_path)
    if not os.path.exists(fn):
        tmpfn = os.path.join(pkg_tmpdir, "model.torch")
        subprocess.run(
            [
                "wget",
                "-O",
                tmpfn,
                "https://figshare.com/ndownloader/files/43423236",
            ],
            check=True,
        )
        shutil.move(tmpfn, fn)
    return relative_path


if __name__ == "__main__":
    sys.exit(main(sys.argv))
