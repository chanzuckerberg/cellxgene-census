import datetime
import itertools
import pickle
import sys
import warnings
from typing import List

import cellxgene_census
import numpy as np
import ontology_mapper
import pandas as pd
import scanpy as sc
import scib_metrics
import tiledbsoma as soma
import yaml
from cellxgene_census.experimental import get_embedding

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    try:
        file = sys.argv[1]
    except IndexError:
        file = "scib-metrics-config.yaml"

    with open(file) as f:
        config = yaml.safe_load(f)

    census_config = config.get("census")
    embedding_config = config.get("embeddings")
    metrics_config = config.get("metrics")

    census_version = census_config.get("version")
    experiment_name = census_config.get("organism")

    embedding_uris_community = embedding_config.get("hosted") or dict()

    # These are embeddings contributed by the community hosted in S3
    # embedding_uris_community = {
    #     "scgpt": f"s3://cellxgene-contrib-public/contrib/cell-census/soma/{CENSUS_VERSION}/CxG-contrib-1/",
    #     "uce": f"s3://cellxgene-contrib-public/contrib/cell-census/soma/{CENSUS_VERSION}/CxG-contrib-2/",
    # }

    # These are embeddings included in the Census data
    embedding_names_census = embedding_config.get("collaboration") or dict()

    embeddings_raw = embedding_config.get("raw") or dict()

    # All embedding names
    embs = list(embedding_uris_community.keys()) + embedding_names_census + list(embeddings_raw.keys())

    print("Embeddings to use: ", embs)

    census = cellxgene_census.open_soma(census_version=census_version)

    def subclass_mapper():
        mapper = ontology_mapper.CellSubclassMapper(map_orphans_to_class=True)
        cell_types = (
            census["census_data"]["homo_sapiens"]
            .obs.read(column_names=["cell_type_ontology_term_id"], value_filter="is_primary_data == True")
            .concat()
            .to_pandas()
        )
        cell_types = cell_types["cell_type_ontology_term_id"].drop_duplicates()
        subclass_dict = {i: mapper.get_label_from_id(mapper.get_top_high_level_term(i)) for i in cell_types}
        return subclass_dict

    def class_mapper():
        mapper = ontology_mapper.CellClassMapper()
        cell_types = (
            census["census_data"]["homo_sapiens"]
            .obs.read(column_names=["cell_type_ontology_term_id"], value_filter="is_primary_data == True")
            .concat()
            .to_pandas()
        )
        cell_types = cell_types["cell_type_ontology_term_id"].drop_duplicates()
        class_dict = {i: mapper.get_label_from_id(mapper.get_top_high_level_term(i)) for i in cell_types}
        return class_dict

    class_dict = class_mapper()
    subclass_dict = subclass_mapper()

    def build_anndata_with_embeddings(
        embedding_uris: dict,
        embedding_names: List[str],
        embeddings_raw: dict,
        coords: List[int] = None,
        obs_value_filter: str = None,
        column_names=dict,
        census_version: str = None,
        experiment_name: str = None,
    ):
        """
        For a given set of Census cell coordinates (soma_joinids)
        fetch embeddings with TileDBSoma and return the corresponding
        AnnData with embeddings slotted in.

        `embedding_uris` is a dict with community embedding names as the keys and S3 URI as the values.
        `embedding_names` is a list with embedding names included in Census.
        `embeddings_raw` are embeddings provided in raw format (npy) on a local drive


        Assume that all embeddings provided are coming from the same experiment.
        """

        with cellxgene_census.open_soma(census_version=census_version) as census:
            print("Getting anndata with Census embeddings: ", embedding_names)

            ad = cellxgene_census.get_anndata(
                census,
                organism=experiment_name,
                measurement_name="RNA",
                obs_value_filter=obs_value_filter,
                obs_coords=coords,
                obsm_layers=embedding_names,
                column_names=column_names,
            )

            obs_soma_joinids = ad.obs["soma_joinid"].to_numpy()

            for key, val in embedding_uris.items():
                print("Getting community embedding:", key)
                embedding_uri = val["uri"]
                ad.obsm[key] = get_embedding(census_version, embedding_uri, obs_soma_joinids)

            # For these, we need to extract the right cells via soma_joinid
            for key, val in embeddings_raw.items():
                print("Getting raw embedding:", key)
                # Alternative approach: set type in the config file
                try:
                    # Assume it's a numpy ndarray
                    emb = np.load(val["uri"])
                    ad.obsm[key] = emb[obs_soma_joinids]
                except Exception:
                    # Assume it's a TileDBSoma URI
                    with soma.open(val["uri"]) as E:
                        embedding_shape = (len(obs_soma_joinids), E.shape[1])
                        embedding = np.full(embedding_shape, np.NaN, dtype=np.float32, order="C")

                        obs_indexer = pd.Index(obs_soma_joinids)
                        for tbl in E.read(coords=(obs_soma_joinids,)).tables():
                            obs_idx = obs_indexer.get_indexer(tbl.column("soma_dim_0").to_numpy())  # type: ignore[no-untyped-call]
                            feat_idx = tbl.column("soma_dim_1").to_numpy()
                            emb = tbl.column("soma_data")

                            indices = obs_idx * E.shape[1] + feat_idx
                            np.put(embedding.reshape(-1), indices, emb)

                        ad.obsm[key] = embedding

        # Embeddings with missing data contain all NaN,
        # so we must find the intersection of non-NaN rows in the fetched embeddings
        # and subset the AnnData accordingly.
        filt = np.ones(ad.shape[0], dtype="bool")
        for key in ad.obsm.keys():
            nan_row_sums = np.sum(np.isnan(ad.obsm[key]), axis=1)
            total_columns = ad.obsm[key].shape[1]
            filt = filt & (nan_row_sums != total_columns)
        ad = ad[filt].copy()

        return ad

    column_names = {
        "obs": ["cell_type_ontology_term_id", "cell_type", "assay", "suspension_type", "dataset_id", "soma_joinid"]
    }
    umap_plot_labels = ["cell_subclass", "cell_class", "cell_type", "dataset_id"]

    block_cell_types = ["native cell", "animal cell", "eukaryotic cell"]

    all_bio = {}
    all_batch = {}

    tissues = metrics_config.get("tissues")

    for tissue in tissues:
        print("Tissue", tissue, " getting Anndata")

        # Getting anddata
        adata_metrics = build_anndata_with_embeddings(
            embedding_uris=embedding_uris_community,
            embedding_names=embedding_names_census,
            embeddings_raw=embeddings_raw,
            obs_value_filter=f"tissue_general == '{tissue}' and is_primary_data == True",
            census_version=census_version,
            experiment_name="homo_sapiens",
            column_names=column_names,
        )

        # Create batch variable
        adata_metrics.obs["batch"] = (
            adata_metrics.obs["assay"] + adata_metrics.obs["dataset_id"] + adata_metrics.obs["suspension_type"]
        )

        # Get cell subclass
        adata_metrics.obs["cell_subclass"] = adata_metrics.obs["cell_type_ontology_term_id"].replace(subclass_dict)
        adata_metrics = adata_metrics[~adata_metrics.obs["cell_subclass"].isna(),]

        # Get cell class
        adata_metrics.obs["cell_class"] = adata_metrics.obs["cell_type_ontology_term_id"].replace(class_dict)
        adata_metrics = adata_metrics[~adata_metrics.obs["cell_class"].isna(),]

        # Remove cells in block list of cell types
        adata_metrics[~adata_metrics.obs["cell_type"].isin(block_cell_types),]

        print("Tissue", tissue, "cells", adata_metrics.n_obs)

        # Calculate neighbors
        for emb_name in embs:
            print(datetime.datetime.now(), "Getting neighbors", emb_name)
            sc.pp.neighbors(adata_metrics, use_rep=emb_name, key_added=emb_name)
            sc.tl.umap(adata_metrics, neighbors_key=emb_name)
            adata_metrics.obsm["X_umap_" + emb_name] = adata_metrics.obsm["X_umap"].copy()
            del adata_metrics.obsm["X_umap"]

        # Save a few UMAPS
        print(datetime.datetime.now(), "Saving UMAP plots")
        for emb_name in embs:
            for label in umap_plot_labels:
                title = "_".join(["UMAP", tissue, emb_name, label])
                sc.pl.embedding(
                    adata_metrics, basis="X_umap_" + emb_name, color=label, title=title, save=title + ".png"
                )

        bio_labels = ["cell_subclass", "cell_class"]
        metric_bio_results = {
            "embedding": [],
            "bio_label": [],
            "leiden_nmi": [],
            "leiden_ari": [],
            "silhouette_label": [],
        }

        batch_labels = ["batch", "assay", "dataset_id", "suspension_type"]
        metric_batch_results = {
            "embedding": [],
            "batch_label": [],
            "silhouette_batch": [],
        }

        for bio_label, emb in itertools.product(bio_labels, embs):
            print("\n\nSTART", bio_label, emb)

            metric_bio_results["embedding"].append(emb)
            metric_bio_results["bio_label"].append(bio_label)

            print(datetime.datetime.now(), "Calculating ARI Leiden")
            this_metric = scib_metrics.nmi_ari_cluster_labels_leiden(
                X=adata_metrics.obsp[emb + "_connectivities"],
                labels=adata_metrics.obs[bio_label],
                optimize_resolution=True,
                resolution=1.0,
                n_jobs=64,
            )
            metric_bio_results["leiden_nmi"].append(this_metric["nmi"])
            metric_bio_results["leiden_ari"].append(this_metric["ari"])

            print(datetime.datetime.now(), "Calculating silhouette labels")

            this_metric = scib_metrics.silhouette_label(
                X=adata_metrics.obsm[emb], labels=adata_metrics.obs[bio_label], rescale=True, chunk_size=512
            )
            metric_bio_results["silhouette_label"].append(this_metric)

        for batch_label, emb in itertools.product(batch_labels, embs):
            print("\n\nSTART", batch_label, emb)

            metric_batch_results["embedding"].append(emb)
            metric_batch_results["batch_label"].append(batch_label)

            print(datetime.datetime.now(), "Calculating silhouette batch")

            this_metric = scib_metrics.silhouette_batch(
                X=adata_metrics.obsm[emb],
                labels=adata_metrics.obs[bio_label],
                batch=adata_metrics.obs[batch_label],
                rescale=True,
                chunk_size=512,
            )
            metric_batch_results["silhouette_batch"].append(this_metric)

        all_bio[tissue] = metric_bio_results
        all_batch[tissue] = metric_batch_results

    with open("metrics_bio.pickle", "wb") as fp:
        pickle.dump(all_bio, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open("metrics_batch.pickle", "wb") as fp:
        pickle.dump(all_batch, fp, protocol=pickle.HIGHEST_PROTOCOL)
