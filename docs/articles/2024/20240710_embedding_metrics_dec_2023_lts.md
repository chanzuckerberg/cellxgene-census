# Benchmarks of single-cell Census models

*Published:* *July 10th, 2024*

*By:* *[Emanuele Bezzi](mailto:ebezzi@chanzuckerberg.com), [Pablo Garcia-Nieto](mailto:pgarcia-nieto@chanzuckerberg.com)*

In 2023, the Census team released a series of cells embeddings (available at the [Census Model page](https://cellxgene.cziscience.com/census-models)) compatible with the [Census LTS version `census_version="2023-12-15"`](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_data_release_info.html#lts-2023-12-15), so that users can access and download for any slice of Census data.

These embeddings were generated via different large-scale models; in this article we present the results of light benchmarking of them. We hope that these benchmarks provide an initial picture to users on, 1) the strength of biological signal captured by these embeddings and, 2) the level of batch correction they exert.

We advise our users to consider these benchmarks as first-pass information and we recommend further benchmarking for a more comprehensive view of the embeddings and for task-oriented applications.

The benchmarks were run on the following embeddings:

- scVI latent spaces from a model trained on all Census data.
- Fine-tuned Geneformer.
- Zero-shot scGPT.
- Zero-shot Universal Cell Embeddings (UCE).

For more details on each model please see the [Census Model page](https://cellxgene.cziscience.com/census-models).

## Accessing the embeddings included in the benchmark

Please the [Census Model page](https://cellxgene.cziscience.com/census-models) for full details. Shortly, you can see the embeddings available for the Census LTS version `census_version="2023-12-15"` using the Census API as follows.

```python
import cellxgene_census.experimental.get_all_available_embeddings
cellxgene_census.experimental.get_all_available_embeddings(census_version="2023-12-15")
```

With the exception of NMF factors, all other human embeddings were included in the benchmarks below. If you would want to access the embeddings for any slice of data you can utilize the parameter `obs_embeddings` from  the`get_anndata()` method of the Census API, for example:

```python
import cellxgene_census
census = cellxgene_census.open_soma(census_version="2023-12-15")
adata = cellxgene_census.get_anndata(
    census,
    organism = "homo_sapiens",
    measurement_name = "RNA",
    obs_value_filter = "tissue_general == 'central nervous system'",
    obs_embeddings = ["scvi"]
)
```

## Benchmarks of Census Embeddings

### About the benchmarks

We executed a series of benchmarks falling into two general types: one to assess the level of biological signal contained in the embeddings, and the second to measure the level of correction for batch effects. In general, the utility of the embeddings increases as a function of these two set of benchmarks.

For each of the type, the benchmarks can be further subdivided by their "mode". A series of metrics assess the embedding space, and the others assess the capacity of the embeddings to predict labels.

The table below shows a breakdown of the benchmarks we used in this report.

<table class="custom-table">
  <thead>
      <tr>
        <th>Type</th>
        <th>Mode</th>
        <th>Metric</th>
        <th>Description</th>
      </tr>
  </thead>
  <tbody>
      <tr>
        <td rowspan="6">Bio-conservation</td>
        <td rowspan="3">Embedding<br>Space</td>
        <td><code>leiden_nmi</code></td>
        <td>Normalized Mutual Information of biological labels and leiden clusters. Described in <a href="https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html">Luecken et al.</a> and implemented in <a href="https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html">scib-metrics.</a></td>
      </tr>
      <tr>
        <td><code>leiden_ari</code></td>
        <td>Adjusted Rand Index of biological labels and leiden clusters. Described in <a href="https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html">Luecken et al.</a> and implemented in <a href="https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html">scib-metrics.</a></td>
      </tr>
      <tr>
        <td><code>silhouette_label</code></td>
        <td>Silhouette score with respect to biological labels. Described in <a href="https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html">Luecken et al.</a> and implemented in <a href="https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.silhouette_label.html">scib-metrics.</a></td>
      </tr>
      <tr>
           <td rowspan="3">Label<br>Classifier</td>
        <td><code>classifier_svm</code></td>
        <td>Accuracy of biological label prediction using a SVM (60/40 train/test split). Implemented <a href="https://github.com/chanzuckerberg/cellxgene-census/blob/f44637ba33567400820407f4f7b9984e52966156/tools/models/metrics/run-scib.py#L36">here</a>.</td>
      </tr>
      <tr>
<td><code>classifier_forest</code></td>
        <td>Accuracy of biological label prediction using a Random Forest classifier (60/40 train/test split). Implemented <a href="https://github.com/chanzuckerberg/cellxgene-census/blob/f44637ba33567400820407f4f7b9984e52966156/tools/models/metrics/run-scib.py#L39">here</a>.</td>
      </tr>
      <tr>
<td><code>classifier_lr</code></td>
        <td>Accuracy of biological label prediction using a Logistic regression classifier (60/40 train/test split). Implemented <a href="https://github.com/chanzuckerberg/cellxgene-census/blob/f44637ba33567400820407f4f7b9984e52966156/tools/models/metrics/run-scib.py#L39">here</a>.</td>
      </tr>
      <tr>
        <td rowspan="5">Batch-correction</td>
        <td rowspan="2">Embedding<br>Space</td>
        <td><code>silhouette_batch</code></td>
        <td>1- silhouette score with respect to biological labels. Described in <a href="https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html">Luecken et al.</a> and implemented in <a href="https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html">scib-metrics.</a></td>
      </tr>
      <tr>
        <td><code>entropy</code></td>
        <td>Average of neighborhood entropy of batch labels per cell. Implemented <a href="https://github.com/chanzuckerberg/cellxgene-census/blob/f44637ba33567400820407f4f7b9984e52966156/tools/models/metrics/run-scib.py#L86">here</a>.</td>
      </tr>
      <tr>
           <td rowspan="3">Label<br>Classifier</td>
        <td><code>classifier_svm</code></td>
        <td>1 - accuracy of batch label prediction using a SVM (60/40 train/test split). Implemented <a href="https://github.com/chanzuckerberg/cellxgene-census/blob/f44637ba33567400820407f4f7b9984e52966156/tools/models/metrics/run-scib.py#L45">here</a>.</td>
      </tr>
      <tr>
<td><code>classifier_forest</code></td>
        <td>1 - accuracy of batch label prediction using a Random Forest classifier (60/40 train/test split). Implemented <a href="https://github.com/chanzuckerberg/cellxgene-census/blob/f44637ba33567400820407f4f7b9984e52966156/tools/models/metrics/run-scib.py#L48">here</a>.</td>
      </tr>
      <tr>
<td><code>classifier_lr</code></td>
        <td>1 - accuracy of batch label prediction using a Logistic regression classifier (60/40 train/test split). Implemented <a href="https://github.com/chanzuckerberg/cellxgene-census/blob/f44637ba33567400820407f4f7b9984e52966156/tools/models/metrics/run-scib.py#L42">here</a>.</td>
      </tr>
  </tbody>
</table>

**Table 1:** List of benchmarks.

### Benchmark results

As reminder the benchmarks were run on the following embeddings:

- scVI latent spaces from a model trained on all Census data.
- Fine-tuned Geneformer.
- Zero-shot scGPT.
- Zero-shot Universal Cell Embeddings (UCE).

#### Summary

The following are averages for all the metrics shown in the following sections.

```{figure} ./20240710_metrics_0_summary.png
:alt: Bio-conservation single-cell Census benchmark
:align: center
:figwidth: 90%

**Figure 1. Summary of all benchmarks.** Numerical averages across the metric types and modes from all bio- and batch-labels across the tissues in this report.
```

#### Bio-conservation

The bio-conservation metrics were run the in following biological labels in a Census cells from Adipose Tissue and Spinal Cord:

- Cell subclass: a higher definition of a cell type with maximum of 73 unique labels, as defined on the CELLxGENE collection page.
- Cell class: an even higher definition of a cell type with a maximum of 22 unique labels, also defined on the CELLxGENE collection page.

```{figure} ./20240710_metrics_1_bio_emb.png
:alt: Bio-conservation single-cell Census benchmark
:align: center
:figwidth: 90%

**Figure 2. Bio-conservation metrics on the embedding space.** Higher values signify better performance, max value for all metrics is 1.
```

```{figure} ./20240710_metrics_2_bio_classifier.png
:alt: Bio-conservation single-cell Census benchmark
:align: center
:figwidth: 90%

**Figure 3. Bio-conservation metrics based on label classifiers.** Values represent label prediction accuracy. Higher values signify better performance, max value for all metrics is 1.
```

#### Batch-correction

The batch-correction metrics were run the in following batch labels in a Census cells from Adipose Tissue and Spinal Cord:

- Assay: the sequencing technology.
- Dataset: the dataset from which the cell originated from.
- Suspension type: cell vs nucleus.
- Batch: the concatenation of values for all of the above.

```{figure} ./20240710_metrics_3_batch_emb.png
:alt: Batch-correction single-cell Census benchmark
:align: center
:figwidth: 90%

**Figure 3. Batch-correction metrics on the embedding space.** Higher values signify better performance, max value for `silhouette_batch` is 1, `entropy` values should only be compared within the tissue/label combination and not across. ```

```{figure} ./20240710_metrics_4_batch_classifier.png
:alt: Batch-correction single-cell Census benchmark
:align: center
:figwidth: 90%

**Figure 4. Batch-correction metrics based on label classifiers.** Values represent **1 - label prediction accuracy**. In theory higher values signify better performance indicating that prediction of batch labels is not accurate. However foundation models may be designed to learn *all* information including technical variation, please refer to the original publications of the models to learn more about them. 
```

## Source data

All data was obtained from the Census API, to fetch the data used in this report you can execute the following in Python. To get the cell subclass and cell class please refer to the [CellxGene Ontology Guide API](https://github.com/chanzuckerberg/cellxgene-ontology-guide/tree/main).

```python
import cellxgene_census

val_filters = {
   "adipose": "tissue_general == 'adipose tissue' and is_primary_data == True",
   "spinal": "tissue_general == 'spinal cord' and is_primary_data == True",
}

embedding_names = ["geneformer", "scgpt", "scvi", "uce"]
embedding_names = ["scvi"]
column_names = {
   "obs": ["cell_type_ontology_term_id", "cell_type", "assay", "suspension_type", "dataset_id", "soma_joinid"]
}

census = cellxgene_census.open_soma(census_version="2023-12-15")

adatas = []
for tissue in val_filters:
    adatas.append(
       cellxgene_census.get_anndata(
           census,
           organism="homo_sapiens",
           measurement_name="RNA",
           obs_value_filter= val_filters[tissue],
           obs_embeddings=embedding_names,
           column_names=column_names,
        )
    )
```

### Batch label counts

The following shows the batch label counts per tissue:

#### Adipose tissue

<table class="custom-table">
  <thead>
      <tr>
        <th>Type</th>
        <th>Label</th>
        <th>Count</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td rowspan="4">Assay</td>
          <td>10x 3' v3</td>
          <td>91947</td>
      </tr>
      <tr>
          <td>10x 5' transcription profiling</td>
          <td>2121</td>
      </tr>
      <tr>
          <td>microwell-seq</td>
          <td>5916</td>
      </tr>
      <tr>
          <td>Smart-seq2</td>
          <td>651</td>
      </tr>
      <tr>
          <td rowspan="2">Suspension Type</td>
          <td>nucleus</td>
          <td>72335</td>
      </tr>
      <tr>
          <td>cell</td>
          <td>23756</td>
      </tr>
      <tr>
          <td rowspan="4">Dataset</td>
          <td>9d8e5dca-03a3-457d-b7fb-844c75735c83</td>
          <td>72335</td>
      </tr>
      <tr>
          <td>53d208b0-2cfd-4366-9866-c3c6114081bc</td>
          <td>20263</td>
      </tr>
      <tr>
          <td>5af90777-6760-4003-9dba-8f945fec6fdf</td>
          <td>2121</td>
      </tr>
      <tr>
          <td>2adb1f8a-a6b1-4909-8ee8-484814e2d4bf</td>
          <td>1372</td>
      </tr>
   </tbody>
</table>

#### Spinal cord

<table class="custom-table">
  <thead>
      <tr>
        <th>Type</th>
        <th>Label</th>
        <th>Count</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td rowspan="2">Assay</td>
          <td>10x 3' v3</td>
          <td>43840</td>
      </tr>
      <tr>
          <td>microwell-seq</td>
          <td>5916</td>
      </tr>
      <tr>
          <td rowspan="2">Suspension Type</td>
          <td>nucleus</td>
          <td>43840</td>
      </tr>
      <tr>
          <td>cell</td>
          <td>5916</td>
      </tr>
      <tr>
          <td rowspan="3">Dataset</td>
          <td>090da8ea-46e8-40df-bffc-1f78e1538d27</td>
          <td>24190</td>
      </tr>
      <tr>
          <td>c05e6940-729c-47bd-a2a6-6ce3730c4919</td>
          <td>19650</td>
      </tr>
      <tr>
          <td>2adb1f8a-a6b1-4909-8ee8-484814e2d4bf</td>
          <td>5916</td>
      </tr>
   </tbody>
</table>
