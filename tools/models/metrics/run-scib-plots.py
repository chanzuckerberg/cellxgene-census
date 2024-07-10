import sys
import yaml
import pickle
import pandas as pd
import plotnine as pt

from typing import List, Dict


def main():
    
    try:
        file = sys.argv[1]
    except IndexError:
        file = "scib-metrics-config-plot.yaml"

    print (file)

    with open(file) as f:
        config = yaml.safe_load(f)

    metrics = load_metrics(
        metrics_file_dict=config["metrics_files"],
    )

    df_metrics, df_summary = wrangle_metrics(
        metrics=metrics,
        bio_metrics_to_plot=config["metrics_plot"]["bio"],
        batch_metrics_to_plot=config["metrics_plot"]["batch"],
        decimal_values=config["decimal_values"],
        normalize_entropy=config["normalize_entropy"],
    )

    suf = config["output_suffix"]

    p = make_headtmap_summary(df_summary)
    p.save(filename=f"{suf}metrics_0_summary.png", dpi=300)
    
    p = make_headtmap(df_metrics.query("metric_mode == 'Embedding space'"), "Bio-conservation")
    p.save(filename=f"{suf}metrics_1_bio_emb.png", dpi=300)
    
    p = make_headtmap(df_metrics.query("metric_mode == 'Label classifier'"), "Bio-conservation")
    p.save(filename=f"{suf}metrics_2_bio_classifier.png", dpi=300)
    
    p = make_headtmap(df_metrics.query("metric_mode == 'Embedding space'"), "Batch-correction", height= 5)
    p.save(filename=f"{suf}metrics_3_batch_emb.png", dpi=300)

    p = make_headtmap(df_metrics.query("metric_mode == 'Label classifier'"), "Batch-correction", height= 5)
    p.save(filename=f"{suf}metrics_4_batch_classifier.png", dpi=300)


def load_metrics(metrics_file_dict: Dict[str, str]):

    all_metrics = {"bio": {}, "batch":{}}
    for tissue in metrics_file_dict:
        metrics_file = metrics_file_dict[tissue]
        with open(metrics_file, "rb") as op:
            metrics = pickle.load(op)

            # Expand classfier metrics into individual keys
            for metric_type in metrics:
                classifier_types = list(metrics[metric_type]['classifier'][0].keys())
                for classifier_type in classifier_types:
                    metrics[metric_type][f"classifier_{classifier_type}"] = []
                    for i in metrics[metric_type]['classifier']:
                        metrics[metric_type][f"classifier_{classifier_type}"].append(
                            i[classifier_type]
                        )
                del metrics[metric_type]['classifier']

            all_metrics["bio"][tissue] = metrics["bio"]
            all_metrics["batch"][tissue] = metrics["batch"]

    return all_metrics


def wrangle_metrics(
    metrics: Dict, 
    bio_metrics_to_plot: List[str], 
    batch_metrics_to_plot: List[str],
    decimal_values: int,
    normalize_entropy: bool,
):

    # Wrangle to pandas wide format 
    df_batch = {}
    df_bio = {}
    for tissue in metrics["bio"]:
        df_batch[tissue] = pd.DataFrame(metrics["batch"][tissue])
        df_batch[tissue]["tissue"] = tissue
        df_bio[tissue] = pd.DataFrame(metrics["bio"][tissue])
        df_bio[tissue]["tissue"] = tissue
    
    df_batch = pd.concat(df_batch, axis=0, ignore_index=True)
    df_bio = pd.concat(df_bio, axis=0, ignore_index=True)
    
    df_batch = df_batch.rename(columns={"batch_label":"label"})
    df_bio = df_bio.rename(columns={"bio_label":"label"})

    if "entropy" in df_batch and normalize_entropy:
        df_batch["entropy"] = df_batch["entropy"] / df_batch["entropy"].max()

    # Wrangle Batch to pandas long format
    df_batch_long = df_batch.rename(columns = {i: "Value_" + i for i in batch_metrics_to_plot})
    df_batch_long = pd.wide_to_long(
        df_batch_long,
        stubnames="Value",
        i=['embedding', 'label', 'tissue'],
        j='metric',
        sep="_",
        suffix='.*',
    ).reset_index()
    df_batch_long["Metric Type"] = "Batch-correction"

    # Wrangle Bio to pandas long format
    df_bio_long = df_bio.rename(columns = {i: "Value_" + i for i in bio_metrics_to_plot})
    df_bio_long = pd.wide_to_long(
        df_bio_long,
        stubnames="Value",
        i=['embedding', 'label', 'tissue'],
        j='metric',
        sep="_",
        suffix='.*',
    ).reset_index()
    df_bio_long["Metric Type"] = "Bio-conservation"

    # Join Bio and Batch metrics, rename columns for readability
    df_long = pd.concat([df_batch_long, df_bio_long])
    df_long["Value"] = round(df_long["Value"], decimal_values)
    df_long = df_long.rename(columns={"embedding": "Embedding"})

    # Add column to classify "metrics mode", classifier vs embedding 
    df_long["metric_mode"] = "Embedding space"
    df_long.loc[df_long["metric"].str.contains("classi"), "metric_mode"] = "Label classifier"

    df_summary = df_long.groupby(["Embedding", "Metric Type", "metric_mode"], as_index=False).mean("Value")
    df_summary["Value"] = round(df_summary["Value"], decimal_values)
    
    return df_long, df_summary

metrics_pt_theme = pt.theme(
    axis_text_x=pt.element_text(rotation=30, hjust=1),
    panel_grid_major=pt.element_blank(),
    panel_grid_minor=pt.element_blank(),
    panel_border=pt.element_blank(),
)


def make_headtmap(df, metric_type, width=9, height=4, theme=metrics_pt_theme):
    return (pt.ggplot(df.query(f"`Metric Type`=='{metric_type}'"))
     + pt.aes(x="Embedding", y="metric",  fill = "Value")
     + pt.geom_tile()
     + pt.facet_grid(f"{'label'}~{'tissue'}")
     + pt.geom_text(pt.aes(label='Value'))
     + pt.scale_fill_gradient(low="#d9e6f2", high="#2d5986")
     + pt.theme_light()
     + pt.theme(figure_size=(width, height))
     + theme
    )


def make_headtmap_summary(df, width=9, height=2.5, theme=metrics_pt_theme):
    return (pt.ggplot(df)
     + pt.aes(x="Embedding", y="Metric Type",  fill = "Value")
     + pt.geom_tile()
     + pt.facet_wrap(f"~{'metric_mode'}", scales="free")
     + pt.geom_text(pt.aes(label='Value'))
     + pt.theme_light()
     + pt.scale_fill_gradient(low="#d9e6f2", high="#2d5986")
     + pt.theme(figure_size=(width, height))
     + theme
    )


if __name__ == "__main__":
    main()