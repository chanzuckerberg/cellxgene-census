#!/usr/bin/env python3
# mypy: ignore-errors

import argparse
import json
import logging
import multiprocessing
import os
import sys
from collections import Counter

import pandas as pd
import yaml
from datasets import Dataset
from geneformer import DataCollatorForCellClassification
from transformers import BertForSequenceClassification, Trainer
from transformers.training_args import TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(module)s [%(levelname)s] %(message)s")
logger = logging.getLogger(os.path.basename(__file__))
NPROC = multiprocessing.cpu_count()


def main(argv):
    args = parse_arguments(argv)
    if os.path.exists(args.model_out):
        logger.error("output directory already exists: " + args.model_out)
        return 1

    with open(args.config) as infile:
        config = yaml.safe_load(infile)
    if args.epochs:
        config["training_args"]["num_train_epochs"] = args.epochs

    logger.info("config: " + str(config))
    logger.info("loading dataset: " + args.dataset)
    dataset = Dataset.load_from_disk(args.dataset)
    logger.info(str(dataset))
    label_to_id, id_to_label, train_dataset, test_dataset = preprocess_dataset(config, dataset)
    logger.info("train_dataset: " + str(train_dataset))
    logger.info("test_dataset: " + str(test_dataset))

    logger.info("loading pretrained model: " + args.model_in)
    model = BertForSequenceClassification.from_pretrained(
        args.model_in, num_labels=len(label_to_id), output_attentions=False, output_hidden_states=False
    ).to("cuda")

    trainer = make_trainer(config, model, train_dataset, test_dataset)
    logger.info("training...")
    trainer.train()

    logger.info("computing final test_dataset predictions...")
    metrics, error_df = tabulate_errors(config, trainer, test_dataset, id_to_label)

    logger.info("saving model to: " + args.model_out)
    trainer.save_model(args.model_out)
    with open(os.path.join(args.model_out, f"{config['label_feature']}_to_label.json"), "w") as outfile:
        json.dump(label_to_id, outfile, indent=2)
    with open(os.path.join(args.model_out, f"label_to_{config['label_feature']}.json"), "w") as outfile:
        json.dump(id_to_label, outfile, indent=2)
    with open(os.path.join(args.model_out, "eval_metrics.json"), "w") as outfile:
        json.dump(metrics, outfile, indent=2)
    error_df.to_csv(os.path.join(args.model_out, "test_errors.tsv"), sep="\t", index=False)

    logger.info("SUCCESS")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Fine-tune Geneformer model")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "finetune-geneformer.config.yml"),
        help="configuration YAML file",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, help="Training epochs (overrides config training_args.num_train_epochs)"
    )
    parser.add_argument("dataset", type=str, help="saved Dataset path")
    parser.add_argument("model_in", type=str, help="path to pretrained Geneformer model")
    parser.add_argument(
        "model_out", type=str, help="output path for fine-tuned model and statistics (mustn't already exist)"
    )

    args = parser.parse_args(argv[1:])

    logger.info("arguments: " + str(vars(args)))
    return args


def preprocess_dataset(config, dataset):
    label_feature = config["label_feature"]
    label_blocklist = config["label_blocklist"]
    label_min_examples = config["label_min_examples"]
    test_size = config["test_size"]

    # find the distinct label values, filter them, and map them onto integer IDs
    label_counts = Counter(dataset[label_feature])
    labels = [
        label
        for label, count in sorted(label_counts.items())
        if label not in label_blocklist and count >= label_min_examples
    ]
    label_to_id = {label: id for id, label in enumerate(labels)}
    id_to_label = {id: label for label, id in label_to_id.items()}
    logger.info(
        f"{len(label_to_id)} classifier labels = "
        f"{len(label_counts)} input labels - {len(label_counts)-len(label_to_id)}"
        " filtered (label_blocklist/label_min_examples)"
    )
    logger.info(
        "examples to filter out: "
        f"{sum(label_counts[label] for label in label_counts if label not in label_to_id)} / {len(dataset)}"
    )

    # filter the dataset to kept labels, add the "label" feature with the integer ID, and split
    datasets = (
        dataset.filter(lambda it: it[label_feature] in label_to_id, num_proc=NPROC)
        .map(lambda it: {"label": label_to_id[it[label_feature]]}, num_proc=NPROC)
        .train_test_split(shuffle=True, test_size=test_size)
    )

    return label_to_id, id_to_label, datasets["train"], datasets["test"]


def make_trainer(config, model, train_dataset, test_dataset):
    all_training_args = {
        **{
            "save_strategy": "no",
            "output_dir": "/tmp/checkpoints",  # unused with save_strategy: no
            "group_by_length": True,
            "length_column_name": "length",
            "disable_tqdm": False,
            "weight_decay": 0.001,
            "do_train": True,
            "do_eval": True,
            "evaluation_strategy": "epoch",
        },
        **config["training_args"],
    }
    logger.info("all training_args: " + str(all_training_args))
    return Trainer(
        model=model,
        args=TrainingArguments(**all_training_args),
        data_collator=DataCollatorForCellClassification(),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=count_label_errors,
    )


def count_label_errors(eval):
    """
    Count the label errors & error rate in the model's predictions.
    """
    pred_labels = eval.predictions.argmax(-1)
    true_labels = eval.label_ids
    assert len(pred_labels) == len(true_labels)
    errors = sum((1 for i, pred_label in enumerate(pred_labels) if pred_label != true_labels[i]))
    return {"label_errors": errors, "label_error_rate": errors / len(true_labels)}


def tabulate_errors(config, trainer, test_dataset, id_to_label):
    # Generate predictions
    eval = trainer.predict(test_dataset)
    pred_labels = eval.predictions.argmax(-1)
    true_labels = test_dataset["label"]

    # Find erroneous examples
    error_indices = [i for i, pred_label in enumerate(pred_labels) if pred_label != true_labels[i]]

    # Create dataframe
    label_feature = config["label_feature"]
    return eval.metrics, pd.DataFrame(
        {
            "soma_joinid": [test_dataset[i]["soma_joinid"] for i in error_indices],
            f"predicted_{label_feature}": [id_to_label[pred_labels[i]] for i in error_indices],
            f"true_{label_feature}": [id_to_label[true_labels[i]] for i in error_indices],
        }
    ).sort_values([f"true_{label_feature}", "soma_joinid"])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
