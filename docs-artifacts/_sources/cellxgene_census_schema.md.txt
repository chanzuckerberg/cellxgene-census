# CZ CELLxGENE Discover Census Schema 

**Version**: 1.0.0.

**Last edited**: Apr, 2023.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED" "MAY", and "OPTIONAL" in this document are to be interpreted as described in [BCP 14](https://tools.ietf.org/html/bcp14), [RFC2119](https://www.rfc-editor.org/rfc/rfc2119.txt), and [RFC8174](https://www.rfc-editor.org/rfc/rfc8174.txt) when, and only when, they appear in all capitals, as shown here.

## Census overview

The CZ CELLxGENE Discover Census, hereafter referred as Census, is a versioned data object and API for most of the single-cell data hosted at [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/). To learn more about the Census visit the `chanzuckerberg/cellxgene-census` [github repository](https://github.com/chanzuckerberg/cellxgene-census)


To better understand this document the reader should be familiar with the [CELLxGENE dataset schema](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md) and [SOMA](https://github.com/single-cell-data/SOMA/blob/main/abstract_specification.md).

## Definitions

The following terms are used throughout this document:

* adata – generic variable name that refers to an [`AnnData`](https://anndata.readthedocs.io/) object.
* CELLxGENE dataset schema – the data schema for h5ad files served by CELLxGENE Discover, for this Census schema: [CELLxGENE dataset schema version is 3.0.X](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md)
* census\_obj – the Census root object, a SOMACollection.
* Census data release – a versioned Census object deposited in a public bucket and accessible by APIs.
* tissue – original tissue annotation.
* tissue\_general – high-level mapping of a tissue, e.g. "Heart" is the tissue_general of "Heart left ventricle" .

## Census Schema versioning

The Census Schema follows [Semver](https://semver.org/) for its versioning:

* Major: any schema changes that make the Census incompatible with the Census API or SOMA API. Examples:
	* Column deletion in Census `obs`
	* Addition of new modality
* Minor: schema additions that are compatible with public API(s) and SOMA. Examples:
	* New column to Census `obs` is added
	* tissue/tissue_general mapping changes
* Patch: schema fixes. Examples:
	* Editorial schema changes


Changes MUST be documented in the schema [Changelog](#changelog) at the end of this document.

Census data releases are versioned separately from the schema.


## Schema

### Data included

All datasets included in the Census MUST be of [CELLxGENE dataset schema version 3.0.X](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md). The following data constraints are imposed on top of the CELLxGENE dataset schema.

#### Species

The Census MUST only contain observations (cells) with an  [`organism_ontology_term_id`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#organism_ontology_term_id) value of either "NCBITaxon:10090" for *Mus musculus* or "NCBITaxon:9606" for *Homo sapiens* MUST be included.

The Census MUST only contain features (genes) with a [`feature_reference`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#feature_reference) value of either "NCBITaxon:10090" for *Mus musculus* or "NCBITaxon:9606" for *Homo sapiens* MUST be included

#### Multi-species data constraints

Per the CELLxGENE dataset schema, [multi-species datasets MAY contain observations (cells) of a given organism and features (genes) of a different one](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#general-requirements), as defined in [`organism_ontology_term_id`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#organism_ontology_term_id) and [`feature_reference`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#feature_reference) respectively. 

For a given multi-species dataset, the table below shows all possible combinations of organisms for both observations and features. For each combination, inclusion criteria for the Census is provided.

<table>
<thead>
  <tr>
    <th>Observations (cells) from</th>
    <th>Features (genes) from</th>
    <th>Inclusion criteria</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>"NCBITaxon:9606" for <i>Homo sapiens</i> AND "NCBITaxon:10090" for <i>Mus musculus</i></td>
    <td>"NCBITaxon:9606" for Homo sapiens</td>
    <td>The Census MUST only contain observations from "NCBITaxon:9606" for <i>Homo sapiens</i></td>
  </tr>
  <tr>
    <td>"NCBITaxon:9606" for <i>Homo sapiens</i> AND "NCBITaxon:10090" for <i>Mus musculus</i></td>
    <td>"NCBITaxon:10090" for <i>Mus musculus</i></td>
    <td>The Census MUST only contain observations from <i>Mus musculus</i></td>
  </tr>
  <tr>
    <td>"NCBITaxon:9606" for <i>Homo sapiens</i> AND "NCBITaxon:10090" for <i>Mus musculus</i></td>
    <td>"NCBITaxon:9606" for <i>Homo sapiens</i> AND "NCBITaxon:10090" for <i>Mus musculus</i></td>
    <td>All observations MUST NOT be included</td>
  </tr>
</tbody>
</table>

#### Assays

Assays are defined in the CELLxGENE dataset schema in [`assay_ontology_term_id`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#assay_ontology_term_id). The Census MUST only contain cells with
an `assay_ontology_term_id` value from the list below:

<table>
<thead>
  <tr>
    <th>ID</th>
    <th>Label</th>
    <th>Notes</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>EFO:0008720</td>
    <td>DroNc-seq</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0008722</td>
    <td>Drop-seq</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0008780</td>
    <td>inDrop</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0008913</td>
    <td>single-cell RNA sequencing</td>
    <td>Refers to "TruDrop" from this <a href="https://cellxgene.cziscience.com/collections/a48f5033-3438-4550-8574-cdff3263fdfd">collection</a> which is not included in EFO at the moment.</td>
  </tr>
  <tr>
    <td>EFO:0008919</td>
    <td>Seq-Well</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0008930</td>
    <td>Smart-seq</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0008931</td>
    <td>Smart-seq2</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0008953</td>
    <td>STRT-seq</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0008995</td>
    <td>10x technology</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0009899</td>
    <td>10x 3' v2</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0009900</td>
    <td>10x 5' v2</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0009901</td>
    <td>10x 3' v1</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0009922</td>
    <td>10x 3' v3</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0010010</td>
    <td>CEL-seq2</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0010183</td>
    <td>single cell library construction</td>
    <td>Refers to "GEXSCOPE" from this <a href="https://cellxgene.cziscience.com/collections/edb893ee-4066-4128-9aec-5eb2b03f8287">collection</a> which is not included in EFO at the moment.</td>
  </tr>
  <tr>
    <td>EFO:0010550</td>
    <td>sci-RNA-seq</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0011025</td>
    <td>10x 5' v1</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0030002</td>
    <td>microwell-seq</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0030003</td>
    <td>10x 3' transcription profiling</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0030004</td>
    <td>10x 5' transcription profiling</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0030019</td>
    <td>Seq-Well S3</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0700003</td>
    <td>BD Rhapsody Whole Transcriptome Analysis</td>
    <td></td>
  </tr>
  <tr>
    <td>EFO:0700004</td>
    <td>BD Rhapsody Targeted mRNA</td>
    <td></td>
  </tr>
</tbody>
</table>

Additional Notes:

* EFO:0030026 "sci-Plex" is not included in spite of being RNA data. This assay has specific cell metadata that is not present in the CELLxGENE dataset schema, without these metadata the RNA data lacks proper context and thus may be misleading to include. 
* EFO:0009920 and EFO:0030062 "Slide-seq" and "Slide-seqV2", respectively, are not included as coverage is low compared to other assays and data lacks context without spatial metadata not included in CELLxGENE dataset schema.

#### Data matrix types

Per the CELLxGENE dataset schema, [all RNA assays MUST include UMI or read counts](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#x-matrix-layers). Author-normalized data layers [as defined in the CELLxGENE dataset schema](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#x-matrix-layers) MUST NOT be included in the Census.

#### Sample types

Only observations (cells) from primary tissue MUST be included in the Census. Thus observations with a [`tissue_ontology_term_id`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#tissue_ontology_term_id) value of  "ontology\_term\_id (organoid)" or "ontology\_term\_id (cell line)" MUST NOT be included.

#### Repeated data

When a cell is represented multiple times in CELLxGENE Discover, only one is marked as the primary cell. This is defined in the CELLxGENE dataset schema under [`is_primary_data`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#is_primary_data). This information MUST be included in the Census cell metadata to enable queries that retrieve datasets (see cell metadata below), and all cells MUST be included in the Census.

### Data encoding and organization

The Census MUST be encoded as a `SOMACollection` which will be referenced  as `census_obj` in the following sections. `census_obj`  MUST have two keys `"census_info"` and `"census_data"` whose contents are defined in the sections below.

#### Census information `census_obj["census_info"]` - `SOMACollection`

A series of summary and metadata tables MUST be included in this `SOMACollection`:

##### Census metadata – `census_obj​​["census_info"]["summary"]` – `SOMADataFrame`

Census metadata MUST be stored as a `SOMADataFrame` with two columns:

<table>
<thead>
  <tr>
    <th>Column</th>
    <th>Encoding</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>label</td>
    <td>string</td>
    <td>Human readable label of metadata variable</td>
  </tr>
  <tr>
    <td>value </td>
    <td>string</td>
    <td>Value associated to metadata variable</td>
  </tr>
</tbody>
</table>

This `SOMADataFrame` MUST have the following rows:
 
1. Census schema version:
	1. label: `"census_schema_version"`
	1. value: Semver schema version.
1. Census build date:
	1. label: `"census_build_date"`
	1. value: The date this Census was built in ISO 8601 date format
1. Dataset schema version:
	1. label: `"dataset_schema_version"`
	1. value: The CELLxGENE Discover schema version of the source H5AD files.
1. Total number of cells included in this Census build:
	1. label: `"total_cell_count"`
	1. value: Cell count
1. Unique number of cells included in this Census build (is_primary_data == True)
	1. label: `"unique_cell_count"`
	1. value: Cell count
1. Number of human donors included in this Census build. Donors are guaranteed to be unique within datasets, not across all Census.
	1. label: `"number_donors_homo_sapiens"`
	1. value: Donor count
1. Number of mouse donors included in this Census build. Donors are guaranteed to be unique within datasets, not across all Census.
	1. label: `"number_donors_mus_musculus"`
	1. value: Donor count

An example of this `SOMADataFrame` is shown below:

<table>
<thead>
  <tr>
    <th>label</th>
    <th>value</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>census_schema_version</td>
    <td>0.0.1</td>
  </tr>
  <tr>
    <td>census_build_date</td>
    <td>2022-11-30</td>
  </tr>
  <tr>
    <td>dataset_schema_version </td>
    <td>3.0.0</td>
  </tr>
  <tr>
    <td>total_cell_count</td>
    <td>10000</td>
  </tr>
  <tr>
    <td>unique_cell_count</td>
    <td>1000</td>
  </tr>
  <tr>
    <td>number_donors_homo_sapiens</td>
    <td>100</td>
  </tr>
  <tr>
    <td>number_donors_mus_musculus</td>
    <td>100</td>
  </tr>
</tbody>
</table>

#### Census table of CELLxGENE Discover datasets – `census_obj["census_info"]["datasets"]` – `SOMADataFrame`

All datasets used to build the Census MUST be included in a table modeled as a `SOMADataFrame`. Each row MUST correspond to an individual dataset with the following columns:

<table>
<thead>
  <tr>
    <th>Column</th>
    <th>Encoding</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>collection_id</td>
    <td>string</td>
    <td rowspan="5">As defined in CELLxGENE Discover <a href="https://api.cellxgene.cziscience.com/curation/ui/">data schema</a> (see &quot;Schemas&quot; section for field definitions)".</td>
  </tr>
  <tr>
    <td>collection_name</td>
    <td>string</td>
  </tr>
  <tr>
    <td>collection_doi</td>
    <td>string</td>
  </tr>
  <tr>
    <td>dataset_id</td>
    <td>string</td>
  </tr>
  <tr>
    <td>dataset_title</td>
    <td>string</td>
  </tr>
  <tr>
    <td>dataset_h5ad_path</td>
    <td>string</td>
    <td>Relative path to the source h5ad file in the Census storage bucket.</td>
  </tr>
  <tr>
    <td>dataset_total_cell_count</td>
    <td>int</td>
    <td>Total number of cells from the dataset included in the Census.</td>
  </tr>
</tbody>
</table>

#### Census summary cell counts  – `census_obj["census_info"]["summary_cell_counts"]` – `SOMADataframe`

Summary cell counts grouped by organism and relevant cell metadata MUST be modeled as a `SOMADataFrame` in `census_obj["census_info"]["summary_cell_counts"]`. Each row of MUST correspond to a combination of organism and metadata variables with the following columns:

<table>
<thead>
  <tr>
    <th>Column</th>
    <th>Encoding</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>organism</td>
    <td>string</td>
    <td>Organism label as shown in NCBITaxon  <code>"Homo sapiens"</code> or <code>"Mus musculus"</code></td>
  </tr>
  <tr>
    <td>category</td>
    <td>string</td>
    <td>Cell metadata used for grouping, one of the following:
        <ul>
          <li><code>all</code> (special case, no grouping)</li>
          <li><code>cell_type</code></li>
          <li><code>assay</code></li>
          <li><code>tissue</code></li>
          <li><code>tissue_general</code> (high-level mapping of a tissue)</li>
          <li><code>disease</code></li>
          <li><code>self_reported_ethnicity</code></li>
          <li><code>sex</code></li>
          <li><code>suspension_type</code></li>
        </ul>
  </tr>
  <tr>
    <td>label</td>
    <td>string</td>
    <td>Label associated to instance of metadata (e.g. <code>"lung"</code> if <code>category</code> is <code>"tissue"</code>). <code>"na"</code> if none.</td>
  </tr>
  <tr>
    <td>ontology_term_id</td>
    <td>string</td>
    <td>ID associated to instance of metadata (e.g. <code>"UBERON:0002048"</code> if category is <code>"tissue"</code>). <code>"na"</code> if none.</td> 
  </tr>
  <tr>
    <td>total_cell_count</td>
    <td>int</td>
    <td>Total number of cell counts for the combination of values of all other fields above.</td>
  </tr>
  <tr>
    <td>unique_cell_count</td>
    <td>int</td>
    <td>Unique number of cells for the combination of values of all other fields above. Unique number of cells refers to the cell count, for this group, when <code><a href="https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#is_primary_data">is_primary_data == True</a></code> </td>
  </tr>
</tbody>
</table>

Example of this `SOMADataFrame`:

<table>
<thead>
  <tr>
    <th>organism</th>
    <th>category</th>
    <th>label</th>
    <th>ontology_term_id</th>
    <th>total_cell_count</th>
    <th>unique_cell_count</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>all</td>
    <td>na</td>
    <td>na</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>cell_type</td>
    <td>cell_type_a</td>
    <td>CL:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>cell_type</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>cell_type</td>
    <td>cell_type_N</td>
    <td>CL:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>assay</td>
    <td>assay_a</td>
    <td>EFO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>assay</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>assay</td>
    <td>assay_N</td>
    <td>EFO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>tissue</td>
    <td>tissue_a</td>
    <td>UBERON:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>tissue</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>tissue</td>
    <td>tissue_N</td>
    <td>UBERON:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>tissue_general</td>
    <td>tissue_general_a</td>
    <td>UBERON:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>tissue_general</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>tissue_general</td>
    <td>tissue_general_N</td>
    <td>UBERON:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>disease</td>
    <td>disease_a</td>
    <td>MONDO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>disease</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>disease</td>
    <td>disease_N</td>
    <td>MONDO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>self_reported_ethnicity</td>
    <td>self_reported_ethnicity_a</td>
    <td>HANCESTRO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>self_reported_ethnicity</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>self_reported_ethnicity</td>
    <td>self_reported_ethnicity_N</td>
    <td>HANCESTRO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>sex</td>
    <td>sex_a</td>
    <td>PATO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>sex</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>sex</td>
    <td>sex_N</td>
    <td>PATO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>suspension_type</td>
    <td>suspension_type_a</td>
    <td>na</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>suspension_type</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>[Homo sapiens|Mus musculus]</td>
    <td>suspension_type</td>
    <td>suspension_type_N</td>
    <td>na</td>
    <td>x</td>
    <td>x</td>
  </tr>
</tbody>
</table>

### Census Data – `census_obj["census_data"][organism]` – `SOMAExperiment`

Data for *Homo sapiens* MUST be stored as a `SOMAExperiment` in `census_obj["homo_sapiens"]`.

Data for *Mus musculus* MUST be stored as a `SOMAExperiment` in `census_obj["mus_musculus"]`.

For each organism the `SOMAExperiment` MUST contain the following:

* Cell metadata – `census_obj["census_data"][organism].obs` – `SOMADataFrame`
* Data  –  `census_obj["census_data"][organism].ms` – `SOMACollection`. This `SOMACollection` MUST only contain one `SOMAMeasurement` in `census_obj["census_data"][organism].ms["RNA"]` with the following:
	* Matrix  data –  `census_obj["census_data"][organism].ms["RNA"].X` – `SOMACollection`. It MUST contain exactly one layer: 
		* Count matrix – `census_obj["census_data"][organism].ms["RNA"].X["raw"]` – `SOMASparseNDArray`
	* Feature metadata – `census_obj["census_data"][organism].ms["RNA"].var` – `SOMAIndexedDataFrame`
	* Feature dataset presence matrix – `census_obj["census_data"][organism].ms["RNA"]["feature_dataset_presence_matrix"]` – `SOMASparseNDArray`

#### Matrix Data, count (raw) matrix – `census_obj["census_data"][organism].ms["RNA"].X["raw"]` – `SOMASparseNDArray`

Per the CELLxGENE dataset schema, [all RNA assays MUST include UMI or read counts](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#x-matrix-layers). These counts MUST be encoded as `float32` in this `SOMASparseNDArray` with a fill value of zero (0), and no explicitly stored zero values.


#### Feature metadata – `census_obj["census_data"][organism].ms["RNA"].var` – `SOMADataFrame`

The Census MUST only contain features with a [`feature_biotype`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#feature_biotype) value of "gene".

The [gene references are pinned](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#required-gene-annotations) as defined in the CELLxGENE dataset schema. 

The following columns MUST be included:

<table>
<thead>
  <tr>
    <th>Column</th>
    <th>Encoding</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>feature_id</td>
    <td>str</td>
    <td>Index of <code>adata.var</code> as defined in CELLxGENE dataset schema</td>
  </tr>
  <tr>
    <td>feature_name</td>
    <td>str</td>
    <td>As defined in CELLxGENE dataset schema </td>
  </tr>
  <tr>
    <td>feature_length</td>
    <td>int</td>
    <td>Gene length in base pairs derived from the <a href="https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#required-gene-annotations">gene reference files from the CELLxGENE dataset schema</a>.</td>
  </tr>
</tbody>
</table>

#### Feature dataset presence matrix – `census_obj["census_data"][organism].ms["RNA"]["feature_dataset_presence_matrix"]` – `SOMASparseNDArray`

In some datasets, there are features not included in the source data. To clarify the difference between features that were not included and features that were not measured, for each `SOMAExperiment` the Census MUST include a presence matrix encoded as a `SOMASparseNDArray`.

For all features included in the Census, the dataset presence matrix MUST indicate what features are included in each dataset of the Census. This information MUST be encoded as a boolean matrix, `True` indicates the feature was included in the dataset, `False` otherwise. This is a two-dimensional matrix and it MUST be `N x M` where `N` is the number of datasets in the `SOMAExperiment` and `M` is the number of features. The matrix is indexed by the `soma_joinid` value of  `census_obj["census_info"]["datasets"]` and `census_obj["census_data"][organism].ms["RNA"].var`.

If the feature has at least one cell with a value greater than zero in the count data matrix X in the dataset of origin, the value MUST be `True`; otherwise, it MUST be `False`.

An example of this matrix is shown below:

<table>
<thead>
  <tr>
    <th></th>
    <th>Feature_1</th>
    <th>…</th>
    <th>Feature_M</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><b>Dataset_soma_joinid_1</b></td>
    <td>[<code>True</code>|<code>False</code>]</td>
    <td>…</td>
    <td>[<code>True</code>|<code>False</code>]</td>
  </tr>
  <tr>
    <td>…</td>
    <td>…</td>
    <td>…</td>
    <td>…</td>
  </tr>
  <tr>
    <td><b> Dataset_soma_joinid_N</b></td>
    <td>[<code>True</code>|<code>False</code>]</td>
    <td>…</td>
    <td>[<code>True</code>|<code>False</code>]</td>
  </tr>
</tbody>
</table>

#### Cell metadata – `census_obj["census_data"][organism].obs` – `SOMADataFrame`

Cell metadata MUST be encoded as a `SOMADataFrame` with the following columns:

<table>
<thead>
  <tr>
    <th>Column</th>
    <th>Encoding</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>dataset_id</td>
    <td>string</td>
    <td>CELLxGENE dataset ID</td>
  </tr>
  <tr>
    <td>tissue_general_ontology_term_id</td>
    <td>string</td>
    <td>High-level tissue UBERON ID as implemented <a href="https://github.com/chanzuckerberg/single-cell-data-portal/blob/9b94ccb0a2e0a8f6182b213aa4852c491f6f6aff/backend/wmg/data/tissue_mapper.py">here</a></td>
  </tr>
  <tr>
    <td>tissue_general</td>
    <td>string</td>
    <td>High-level tissue label as implemented <a href="https://github.com/chanzuckerberg/single-cell-data-portal/blob/9b94ccb0a2e0a8f6182b213aa4852c491f6f6aff/backend/wmg/data/tissue_mapper.py">here</a></td>
  </tr>
  <tr>
    <td>assay_ontology_term_id</td>
    <td colspan="2" rowspan="17">As defined in CELLxGENE dataset schema</td>
  </tr>
  <tr>
    <td>assay</td>
  </tr>
  <tr>
    <td>cell_type_ontology_term_id</td>
  </tr>
  <tr>
    <td>cell_type</td>
  </tr>
  <tr>
    <td>development_stage_ontology_term_id</td>
  </tr>
  <tr>
    <td>development_stage</td>
  </tr>
  <tr>
    <td>disease_ontology_term_id</td>
  </tr>
  <tr>
    <td>disease</td>
  </tr>
  <tr>
    <td>donor_id</td>
  </tr>
  <tr>
    <td>is_primary_data</td>
  </tr>
  <tr>
    <td>self_reported_ethnicity_ontology_term_id</td>
  </tr>
  <tr>
    <td>self_reported_ethnicity</td>
  </tr>
  <tr>
    <td>sex_ontology_term_id</td>
  </tr>
  <tr>
    <td>sex</td>
  </tr>
  <tr>
    <td>suspension_type</td>
  </tr>
  <tr>
    <td>tissue_ontology_term_id</td>
  </tr>
  <tr>
    <td>tissue</td>
  </tr>
</tbody>
</table>


## Changelog

### Version 1.0.0
* Updates text to reflect official name: CZ CELLxGENE Discover Census.
* Updates `census["census_info"]["summary"]` to reflect official name in the column `label`:
   * From `"cell_census_build_date"` to `"census_build_date"`.
   * From `"cell_census_schema_version"` to `"census_schema_version"`.
* Adds the following row to `census["census_info"]["summary"]`:
   * `"dataset_schema_version"`


### Version 0.1.1
* Adds clarifying text for "Feature Dataset Presence Matrix"

### Version 0.1.0
* The "Dataset Presence Matrix" was renamed to "Feature Dataset Presence Matrix" and moved from  `census_obj["census_data"][organism].ms["RNA"].varp["dataset_presence_matrix"]`  to `census_obj["census_data"][organism].ms["RNA"]["feature_dataset_presence_matrix"]`.
* Editorial: changes all double quotes in the schema to ASCII quotes 0x22.

### Version 0.0.1
* Initial Census schema is published.

