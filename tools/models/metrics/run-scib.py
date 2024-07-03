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

import numpy as np
import scipy as sp
import cellxgene_census
import functools

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

class CensusClassifierMetrics:

    def __init__(self):
        self._default_metric = "accuracy"

    def lr_labels(self, X, labels, metric = None):
        return self._base_accuracy(X, labels, LogisticRegression, metric=metric)

    def svm_svc_labels(self, X, labels, metric = None):
        return self._base_accuracy(X, labels, svm.SVC, metric=metric)

    def random_forest_labels(self, X, labels, metric = None, n_jobs=8):
        return self._base_accuracy(X, labels, RandomForestClassifier, metric=metric, n_jobs=n_jobs)

    def lr_batch(self, X, batch, metric = None):
        return 1-self._base_accuracy(X, batch, LogisticRegression, metric=metric)

    def svm_svc_batch(self, X, batch, metric = None):
        return 1-self._base_accuracy(X, batch, svm.SVC, metric=metric)

    def random_forest_batch(self, X, batch, metric = None, n_jobs=8):
        return 1-self._base_accuracy(X, batch, RandomForestClassifier, metric=metric, n_jobs=n_jobs)

    def _base_accuracy(self, X, y, model, metric, test_size=0.4, **kwargs):
        """
        Train LogisticRegression on X with labels y and return classifier accuracy score
        """
        y_encoded = LabelEncoder().fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        model = model(**kwargs).fit(X_train, y_train)

        if metric == None:
            metric = self._default_metric
        
        if metric == "roc_auc":  
            #return y_test
            #return model.predict_proba(X_test)
            return roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovo", average="macro")
        elif metric == "accuracy":
            return accuracy_score(y_test, model.predict(X_test))
        else:
            raise ValueError("Only {'accuracy', 'roc_auc'} are supported as a metric")
        
def safelog(a):
    return np.log(a, out=np.zeros_like(a), where=(a!=0))

def nearest_neighbors_hnsw(x, ef=200, M=48, n_neighbors = 100):
    import hnswlib
    labels = np.arange(x.shape[0])
    p = hnswlib.Index(space = 'l2', dim = x.shape[1])
    p.init_index(max_elements = x.shape[0], ef_construction = ef, M = M)
    p.add_items(x, labels)
    p.set_ef(ef)
    idx, dist = p.knn_query(x, k = n_neighbors)
    return idx,dist

def compute_entropy_per_cell(adata, obsm_key):

    batch_keys = ["dataset_id", "assay", "suspension_type"]
    adata.obs["batch"] = functools.reduce(lambda a, b: a+b, [adata.obs[c].astype(str) for c in batch_keys])

    indices, dist = nearest_neighbors_hnsw(adata.obsm[obsm_key], n_neighbors = 200)

    BATCH_KEY = 'batch'

    batch_labels = np.array(list(adata.obs[BATCH_KEY]))
    unique_batch_labels = np.unique(batch_labels)

    indices_batch = batch_labels[indices]

    label_counts_per_cell = np.vstack([(indices_batch == label).sum(1) for label in unique_batch_labels]).T
    label_counts_per_cell_normed = label_counts_per_cell / label_counts_per_cell.sum(1)[:,None]
    return (-label_counts_per_cell_normed*safelog(label_counts_per_cell_normed)).sum(1)

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

    # These are embeddings hosted in the Census
    embeddings_census = embedding_config.get("census") or []

    # Raw embeddings (external)
    embeddings_raw = embedding_config.get("raw") or dict()

    # All embedding names
    embs = list(embeddings_census) + list(embeddings_raw.keys())

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
                obs_embeddings=embedding_names,
                column_names=column_names,
            )

            obs_soma_joinids = ad.obs["soma_joinid"].to_numpy()

            # For these, we need to extract the right cells via soma_joinid
            for key, val in embeddings_raw.items():
                print("Getting raw embedding:", key)
                # Alternative approach: set type in the config file
                try:
                    # Assume it's a numpy ndarray
                    emb = np.load(val["uri"])
                    emb_idx = np.load(val["idx"])
                    obs_indexer = pd.Index(emb_idx)
                    idx = obs_indexer.get_indexer(obs_soma_joinids)
                    ad.obsm[key] = emb[idx]
                except Exception:
                    from scipy.sparse import vstack
                    # Assume it's a TileDBSoma URI
                    all_embs = []
                    with soma.open(val["uri"]) as E:
                        for mat in E.read(coords=(obs_soma_joinids,)).blockwise(axis=0).scipy():
                            all_embs.append(mat[0])
                        ad.obsm[key] = vstack(all_embs).toarray()
                        print("DIM:", ad.obsm[key].shape)

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

    block_cell_types = ["native cell", "animal cell", "eukaryotic cell", "unknown"]

    all_bio = {}
    all_batch = {}

    tissues = metrics_config.get("tissues")

    bio_metrics = metrics_config["bio"]
    batch_metrics = metrics_config["batch"]

    for tissue_node in tissues:

        tissue = tissue_node["name"]
        query = tissue_node.get("query") or f"tissue_general == '{tissue}' and is_primary_data == True"

        print("Tissue", tissue, " getting Anndata")

        # Getting anddata
        adata_metrics = build_anndata_with_embeddings(
            embedding_names=embeddings_census,
            embeddings_raw=embeddings_raw,
            obs_value_filter=query,
            census_version=census_version,
            experiment_name="homo_sapiens",
            column_names=column_names,
        )

        for column in adata_metrics.obs.columns:
            if adata_metrics.obs[column].dtype.name == "category":
                adata_metrics.obs[column] = adata_metrics.obs[column].astype(str)

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
            # Only necessary
            if "ilisi_knn_batch" in metrics_config["batch"]:
                sc.pp.neighbors(adata_metrics, n_neighbors=90, use_rep=emb_name, key_added=emb_name + "_90")
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
        batch_labels = ["batch", "assay", "dataset_id", "suspension_type"]

        # Initialize results
        metric_bio_results = {
            "embedding": [],
            "bio_label": [],
        }
        metric_batch_results = {
            "embedding": [],
            "batch_label": [],
        }

        for metric in bio_metrics:
            metric_bio_results[metric] = []

        for metric in batch_metrics:
            metric_batch_results[metric] = []

        # Calculate metrics
        for bio_label, emb in itertools.product(bio_labels, embs):
            print("\n\nSTART", bio_label, emb)

            metric_bio_results["embedding"].append(emb)
            metric_bio_results["bio_label"].append(bio_label)

            print(datetime.datetime.now(), "Calculating ARI Leiden")

            class NN:
                def __init__(self, conn):
                    self.knn_graph_connectivities = conn

            X = NN(adata_metrics.obsp[emb + "_connectivities"])

            if "leiden_nmi" in bio_metrics and "leiden_ari" in bio_metrics:
                this_metric = scib_metrics.nmi_ari_cluster_labels_leiden(
                    X=X,
                    labels=adata_metrics.obs[bio_label],
                    optimize_resolution=True,
                    resolution=1.0,
                    n_jobs=64,
                )
                metric_bio_results["leiden_nmi"].append(this_metric["nmi"])
                metric_bio_results["leiden_ari"].append(this_metric["ari"])

            if "silhouette_label" in bio_metrics:
                print(datetime.datetime.now(), "Calculating silhouette labels")

                this_metric = scib_metrics.silhouette_label(
                    X=adata_metrics.obsm[emb], labels=adata_metrics.obs[bio_label], rescale=True, chunk_size=512
                )
                metric_bio_results["silhouette_label"].append(this_metric)

            if "classifier" in bio_metrics:
                metrics = CensusClassifierMetrics()

                m1 = metrics.lr_labels(X=adata_metrics.obsm[emb], labels = adata_metrics.obs["cell_type"])
                m2 = metrics.svm_svc_labels(X=adata_metrics.obsm[emb], labels = adata_metrics.obs["cell_type"])
                m3 = metrics.random_forest_labels(X=adata_metrics.obsm[emb], labels = adata_metrics.obs["cell_type"])

                metric_bio_results["classifier"].append({"lr": m1, "svm": m2, "random_forest": m3})


        for batch_label, emb in itertools.product(batch_labels, embs):
            print("\n\nSTART", batch_label, emb)

            metric_batch_results["embedding"].append(emb)
            metric_batch_results["batch_label"].append(batch_label)

            if "silhouette_batch" in batch_metrics:
                print(datetime.datetime.now(), "Calculating silhouette batch")

                this_metric = scib_metrics.silhouette_batch(
                    X=adata_metrics.obsm[emb],
                    labels=adata_metrics.obs[bio_label],
                    batch=adata_metrics.obs[batch_label],
                    rescale=True,
                    chunk_size=512,
                )
                metric_batch_results["silhouette_batch"].append(this_metric)

            if "ilisi_knn_batch" in batch_metrics:
                print(datetime.datetime.now(), "Calculating ilisi knn batch")

                ilisi_metric = scib_metrics.ilisi_knn(
                    X=adata_metrics.obsp[f"{emb}_90_distances"],
                    batches=adata_metrics.obs[batch_label],
                    scale=True,
                )

                metric_batch_results["ilisi_knn_batch"].append(ilisi_metric)

            if "classifier" in batch_metrics:
                metrics = CensusClassifierMetrics()

                m4 = metrics.lr_batch(X=adata_metrics.obsm[emb], batch = adata_metrics.obs[batch_label])
                m5 = metrics.random_forest_batch(X=adata_metrics.obsm[emb], batch = adata_metrics.obs[batch_label])
                m6 = metrics.svm_svc_batch(X=adata_metrics.obsm[emb], batch = adata_metrics.obs[batch_label])
                metric_batch_results["classifier"].append({"lr": m4, "random_forest": m5, "svm": m6})

            if "entropy" in batch_metrics:
                print(datetime.datetime.now(), "Calculating entropy")

                entropy = compute_entropy_per_cell(adata_metrics, emb)
                e_mean = entropy.mean()
                metric_batch_results["entropy"].append(e_mean)

        all_bio[tissue] = metric_bio_results
        all_batch[tissue] = metric_batch_results

    with open("metrics.pickle", "wb") as fp:
        pickle.dump({"all_bio": all_bio, "all_batch": all_batch}, fp, protocol=pickle.HIGHEST_PROTOCOL)
