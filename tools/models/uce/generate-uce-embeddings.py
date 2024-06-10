#!/usr/bin/env python3
# mypy: ignore-errors

import os
import sys
import shutil
import anndata
import logging
import argparse
import subprocess
import tiledbsoma
import cellxgene_census
import boto3
import tiledb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s [%(levelname)s] %(message)s")
logger = logging.getLogger(os.path.basename(__file__))

def main(argv):
    args = parse_arguments(argv)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.output_dir_census):
        os.makedirs(args.output_dir_census)

    tiledbsoma_context = None
    if args.tiledbsoma:

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

        with tiledbsoma.SparseNDArray.open(args.output_file, "r", context=tiledbsoma_context):
            # TODO: verify schema compatibility
            pass

    # open human census
    logger.info("Generating anndata slice from Census")
    dataset_filename = f"anndata_uce_{args.part}.h5ad"
    dataset_path = (os.path.join(args.output_dir_census, dataset_filename))
    with cellxgene_census.open_soma(census_version=args.census_version) as census:

        # select the cell id's to include
        coords = get_soma_joinid_slice(census["census_data"]["homo_sapiens"], args.part, args.parts)

        adata_census = cellxgene_census.get_anndata(
            census,
            "homo_sapiens",
            "RNA",
            obs_coords=coords,
            column_names={"obs": ["soma_joinid"]},
        )

        adata_census.var_names = adata_census.var["feature_name"]
        adata_census.write_h5ad(dataset_path)

    # Get 33L model
    logger.info("Zero-shot through UCE")
    model_path = uce_33l_model_file("./", "./UCE/")
    uce_dir = "./UCE/"

    if not os.path.exists(os.path.join(uce_dir, dataset_filename)):
        os.makedirs(os.path.join(uce_dir, dataset_filename))

    dataset_path_uce = uce(
        dataset_path,
        uce_dir=uce_dir,
        relative_work_dir=dataset_filename,
        uce_33l_model_file=model_path,
        emb_dim=args.emb_dim,
    )
    
    # Move adata to final location and clean up other output files
    shutil.move(os.path.join(uce_dir, dataset_path_uce), os.path.join(args.output_dir, dataset_filename))
    shutil.rmtree(os.path.join(uce_dir, dataset_filename))
    
    logger.info("Writing embeddings")
    adata = anndata.read_h5ad(os.path.join(args.output_dir, dataset_filename))

    if args.tiledbsoma:
        import numpy as np
        import pyarrow as pa
        from scipy.sparse import coo_matrix

        # NOTE: embs_df has columns -named- 0, 1, ..., 511 as well as the requested features.
        embedding_dim = adata.obsm["X_uce"].shape[1]
        logger.info(f"writing to tiledbsoma.SparseNDArray at {args.output_file}, embedding_dim={embedding_dim}...")
        with tiledbsoma.SparseNDArray.open(args.output_file, "w", context=tiledbsoma_context) as array:
            dim0 = np.repeat(adata.obs["soma_joinid"].values, embedding_dim)
            dim1 = np.tile(np.arange(embedding_dim), len(adata))
            data = adata.obsm["X_uce"].flatten()
            array.write(pa.SparseCOOTensor.from_scipy(coo_matrix((data, (dim0, dim1)))))
    else:
        # TODO implement to csv, for Census LTS this is not needed. Leaving it as an option for
        # consistency with Geneformer pipeline.
        raise NotImplementedError("only tiledbsoma output is supported, please add --tiledbsoma flag to call.")
    
    logger.info("SUCCESS")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Generate UCE embeddings for an anndata")
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="process only one shard of the data (zero-based index)",
        required=True,
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=1000,
        help="Number of total data partitions for Census, effectively number of H5AD files that will be created.",
        required = True,
    )
    parser.add_argument(
        "-v",
        "--census-version",
        type=str,
        default="latest",
        help='Census release to query (default: "latest")',
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=1280,
        help="number of columns for array",
    )
    parser.add_argument(
        "--tiledbsoma",
        action="store_true",
        help="output_file is URI to an existing tiledbsoma.SparseNDArray to write into (instead of TSV file)",
    )
    parser.add_argument(
        "--output-dir-census",
        type=str,
        help="output directory for resulting anndatas from Census without UCE embeddings with format\
        'anndata_uce_{part}.h5ad'",
        required=True,
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="output directory for resulting anndatas with UCE embeddings",
        required=True,
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="URI location of destination tiledbsoma array, other formats not supported",
    )

    args = parser.parse_args(argv[1:])

    if not (args.part >= 0 and args.parts is not None and args.parts > args.part):
        parser.error("--part must be nonnegative and less than --parts")

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


def uce(h5ad, uce_dir, relative_work_dir, uce_33l_model_file, emb_dim, args=None):
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
            "--output_dim",
            str(emb_dim),
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
