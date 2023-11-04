#!/usr/bin/env python3
# mypy: ignore-errors

import argparse
import logging
import os
import sys
import tempfile

import geneformer
from datasets import Dataset, disable_progress_bar
from transformers import BertConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s [%(levelname)s] %(message)s")
logger = logging.getLogger(os.path.basename(__file__))
disable_progress_bar()


def main(argv):
    args = parse_arguments(argv)

    num_classes = 0
    if args.model_type != "Pretrained":
        # classifier model: detect the number of classes from the model configuration (without
        # having to load in the whole model redundantly with EmbExtractor below)
        num_classes = BertConfig.from_pretrained(args.model).num_labels
        logger.info(f"detected {num_classes} labels in {args.model_type} model {args.model}")

    with tempfile.TemporaryDirectory() as scratch_dir:
        # prepare the dataset, taking only one shard of it if so instructed
        dataset_path = prepare_dataset(args.dataset, args.part, args.parts, scratch_dir)
        logger.info("Extracting embeddings...")
        extractor = geneformer.EmbExtractor(
            model_type=args.model_type,
            num_classes=num_classes,
            max_ncells=None,
            emb_layer=args.emb_layer,
            emb_label=args.features,
            forward_batch_size=args.batch_size,
        )
        # NOTE: EmbExtractor will use only one GPU
        #       see https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/emb_extractor.py
        embs_df = extractor.extract_embs(
            model_directory=args.model,
            input_data_file=dataset_path,
            # the method always writes out a .csv file which we discard (since it also returns the
            # embeddings data frame)
            output_directory=scratch_dir,
            output_prefix="embs",
        )

        logger.info(f"writing {args.outfile}...")
        # reorder embs_df columns and write to TSV
        # TODO: determine final output format
        cols = embs_df.columns.tolist()
        emb_cols = [col for col in cols if isinstance(col, int)]
        anno_cols = [col for col in cols if not isinstance(col, int)]
        embs_df = embs_df[anno_cols + emb_cols].set_index("soma_joinid").sort_index()
        embs_df.to_csv(args.outfile, sep="\t", header=True, index=True, index_label="soma_joinid")

        logger.info("SUCCESS")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Generate fine-tuned Geneformer embeddings for given cells dataset")
    parser.add_argument("model", type=str, help="path to model")
    parser.add_argument("dataset", type=str, help="saved cell Dataset path")
    parser.add_argument("outfile", type=str, help="output filename")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("CellClassifier", "Pretrained"),
        default="CellClassifier",
        help="model type (Pretrained or default CellClassifier)",
    )
    parser.add_argument(
        "--emb-layer", type=int, choices=(-1, 0), default=-1, help="desired embedding layer (0 or default -1)"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="soma_joinid,cell_type,cell_type_ontology_term_id,cell_subclass,cell_subclass_ontology_term_id",
        help="dataset features to copy into output dataframe (comma-separated)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--part", type=int, help="process only one shard of the data (zero-based index)")
    parser.add_argument("--parts", type=int, help="required with --part")

    args = parser.parse_args(argv[1:])

    if args.features:
        args.features = [s.strip() for s in args.features.split(",")]

    if args.part is not None:
        if not (args.part >= 0 and args.parts is not None and args.parts > args.part):
            parser.error("--part must be nonnegative and less than --parts")

    logger.info("arguments: " + str(vars(args)))
    return args


def prepare_dataset(dataset_dir, part, parts, spool_dir):
    dataset = Dataset.load_from_disk(dataset_dir)
    logger.info(f"dataset (full): {dataset}")
    if part is None:
        return dataset_dir
    dataset = dataset.shard(num_shards=parts, index=part, contiguous=True)
    # spool the desired part of the dataset (since EmbExtractor takes a directory)
    logger.info(f"dataset part: {dataset}")
    part_dir = os.path.join(spool_dir, "dataset_part")
    dataset.save_to_disk(part_dir)
    return part_dir


if __name__ == "__main__":
    sys.exit(main(sys.argv))
