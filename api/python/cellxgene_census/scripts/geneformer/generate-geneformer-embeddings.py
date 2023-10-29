#!/usr/bin/env python3
# mypy: ignore-errors

import argparse
import json
import logging
import multiprocessing
import os
import sys

from datasets import Dataset
from geneformer import DataCollatorForCellClassification
from transformers import BertForSequenceClassification, Trainer
from transformers.training_args import TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s [%(levelname)s] %(message)s")
logger = logging.getLogger(os.path.basename(__file__))
NPROC = multiprocessing.cpu_count()


def main(argv):
    args = parse_arguments(argv)
    label_names, model = load_model(args.model, args.label_feature)
    dataset = load_dataset(args.dataset, args.part, args.parts)

    training_args = TrainingArguments(
        save_strategy="no",
        output_dir="/tmp/checkpoints",  # unused with save_strategy: no
        per_device_eval_batch_size=args.batch_size,
        group_by_length=True,
        length_column_name="length",
    )
    trainer = Trainer(model=model, data_collator=DataCollatorForCellClassification(), args=training_args)

    logger.info("running inference...")
    eval = trainer.predict(dataset)
    pred_labels = eval.predictions.argmax(-1)
    true_labels = dataset[args.label_feature] if args.label_feature in dataset.features else None

    # TODO: actually extract embeddings !

    logger.info(f"writing {args.outfile}...")
    with open(args.outfile, "w") as outfile:
        header = ["soma_joinid", f"predicted_{args.label_feature}"]
        if true_labels:
            header.append(args.label_feature)
        print("\t".join(header), file=outfile)
        for i, it in enumerate(dataset):
            row = [it["soma_joinid"], label_names[str(pred_labels[i])]]
            if true_labels:
                row.append(true_labels[i])
            print("\t".join(row), file=outfile)

    logger.info("SUCCESS")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Generate fine-tuned Geneformer embeddings for given cells dataset")
    parser.add_argument("model", type=str, help="path to fine-tuned Geneformer model")
    parser.add_argument("dataset", type=str, help="saved cell Dataset path")
    parser.add_argument("outfile", type=str, help="output filename")
    parser.add_argument(
        "--label-feature",
        type=str,
        default="cell_subclass",
        help="feature the model was fine-tuned to predict (default: cell_subclass)",
    )
    parser.add_argument("--part", type=int, help="process only cells with soma_joinid %% parts == part")
    parser.add_argument("--parts", type=int, help="required with --part")
    parser.add_argument("--batch-size", type=int, default=16, help="prediction batch size")

    args = parser.parse_args(argv[1:])

    if args.part is not None:
        if not (args.part >= 0 and args.parts is not None and args.parts > args.part):
            parser.error("--part must be nonnegative and less than --parts")

    logger.info("arguments: " + str(vars(args)))
    return args


def load_model(model_dir, label_feature):
    """
    Load the model and the mapping from label number to cell_subclass (or whatever feature was used)
    """
    label_names_file = os.path.join(model_dir, f"label_to_{label_feature}.json")
    try:
        with open(label_names_file) as infile:
            label_names = json.load(infile)
    except FileNotFoundError:
        raise FileNotFoundError(f"expected to find {label_names_file}; check --label-feature") from None
    logger.info("labels: " + " ".join(label_names.values()))
    model = BertForSequenceClassification.from_pretrained(
        model_dir, num_labels=len(label_names), output_attentions=False, output_hidden_states=False
    ).to("cuda")
    return label_names, model


def load_dataset(dataset_dir, part, parts):
    dataset = Dataset.load_from_disk(dataset_dir)
    logger.info(f"dataset (full): {dataset}")
    if part is not None:
        dataset = dataset.filter(lambda it: it["soma_joinid"] % parts == part, num_proc=NPROC)
        logger.info(f"dataset part: {dataset}")
    return dataset


if __name__ == "__main__":
    sys.exit(main(sys.argv))
