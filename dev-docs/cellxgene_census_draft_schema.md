# CZ CELLxGENE Discover Census Schema

Document Status: _Drafting_

**Version**: 2.4.0

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED" "MAY", and "OPTIONAL" in this document are to be interpreted as described in [BCP 14](https://tools.ietf.org/html/bcp14), [RFC2119](https://www.rfc-editor.org/rfc/rfc2119.txt), and [RFC8174](https://www.rfc-editor.org/rfc/rfc8174.txt) when, and only when, they appear in all capitals, as shown here.

## Census overview

The [CZ CELLxGENE Discover Census](https://chanzuckerberg.github.io/cellxgene-census/) is a versioned data object and API for most of the single-cell data hosted at [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/). It is referred to throughout this document as the Census.

The reader should be familiar with the [CELLxGENE Discover dataset schema](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md) and the [SOMA (“stack of matrices, annotated”)](https://github.com/single-cell-data/SOMA/blob/main/abstract_specification.md) specification.

## Definitions

The following terms are used throughout this document:

**EDITORIAL NOTE: Review and remove as needed.**

* adata – generic variable name that refers to an [`AnnData`](https://anndata.readthedocs.io/) object.
* CELLxGENE Discover dataset schema – the data schema for h5ad files served by CELLxGENE Discover, for this Census schema: [CELLxGENE dataset schema version is 7.0.0](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md)
* census\_obj – the Census root object, a [SOMACollection](https://github.com/single-cell-data/SOMA/blob/main/abstract_specification.md#somacollection).
* Census data release – a versioned Census object deposited in a public bucket and accessible by APIs.
* tissue – original tissue annotation.
* tissue\_general – high-level mapping of a tissue, e.g. "Heart" is the tissue_general of "Heart left ventricle" .
* scene – a collection of spatially resolved data that can be registered to a single coordinate space.

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

All datasets included in the Census MUST be of [CELLxGENE Discover dataset schema version 7.0.0](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md). The following data constraints are imposed on top of the CELLxGENE Discover dataset schema.

#### Organisms

The CELLxGENE Discover dataset schema requires one [`organism_ontology_term_id`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#organism_ontology_term_id) per dataset. Each [`feature_reference`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#feature_reference) MUST contain a matching value or:
<ul>
  <li>
    <a href="https://www.ebi.ac.uk/ols4/ontologies/ncbitaxon/classes?obo_id=NCBITaxon%3A2697049"><code>"NCBITaxon:2697049"</code></a> for <i>SARS-CoV-2</i>
  </li>

  <li>
    <a href="https://www.ebi.ac.uk/ols4/ontologies/ncbitaxon/classes?obo_id=NCBITaxon%3A32630"><code>"NCBITaxon:32630"</code></a> for <i>ERCC Spike-Ins</i>
  </li>
</ul><br>

The Census MUST only include observations (cells) corresponding to the following values for [`organism_ontology_term_id`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#organism_ontology_term_id) and MUST only include features corresponding to the following values for [`feature_reference`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#feature_reference):

<table>
  <thead>
    <tr>
      <th>Value</th>
      <th>for Organism</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <a href="https://www.ebi.ac.uk/ols4/ontologies/ncbitaxon/classes?obo_id=NCBITaxon%3A9483"><code>"NCBITaxon:9483"</code></a>
      </td>
      <td><i>Callithrix jacchus</i></td>
    </tr>
    <tr>
      <td>
        <a href="https://www.ebi.ac.uk/ols4/ontologies/ncbitaxon/classes?obo_id=NCBITaxon%3A9606"><code>"NCBITaxon:9606"</code></a>
      </td>
      <td><i>Homo sapiens</i></td>
    </tr>
    <tr>
      <td>
        <a href="https://www.ebi.ac.uk/ols4/ontologies/ncbitaxon/classes?obo_id=NCBITaxon%3A9544"><code>"NCBITaxon:9544"</code></a>
        </td>
      <td><i>Macaca mulatta</i></td>
    </tr>
    <tr>
      <td>
        <a href="https://www.ebi.ac.uk/ols4/ontologies/ncbitaxon/classes?obo_id=NCBITaxon%3A10090"><code>"NCBITaxon:10090"</code></a>
      </td>
      <td><i>Mus musculus</i></td>
    </tr>
    <tr>
      <td>
          <a href="https://www.ebi.ac.uk/ols4/ontologies/ncbitaxon/classes?obo_id=NCBITaxon%3A9598"><code>"NCBITaxon:9598"</code></a>
      </td>
      <td><i>Pan troglodytes</td>
    </tr>
  </tbody>
</table><br>

The following values for [`feature_reference`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#feature_reference) MUST NOT be included:

<table>
  <thead>
    <tr>
      <th>Value</th>
      <th>for Organism</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <a href="https://www.ebi.ac.uk/ols4/ontologies/ncbitaxon/classes?obo_id=NCBITaxon%3A2697049"><code>"NCBITaxon:2697049"</code></a>
      </td>
      <td><i>SARS-CoV-2</i></td>
    </tr>
    <tr>
      <td>
        <a href="https://www.ebi.ac.uk/ols4/ontologies/ncbitaxon/classes?obo_id=NCBITaxon%3A32630"><code>"NCBITaxon:32630"</code></a>
      </td>
      <td><i>ERCC Spike-Ins</i></td>
    </tr>
  </tbody>
</table>

#### Assays

CELLxGENE Discover requirements for assays are defined in [`assay_ontology_term_id`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#assay_ontology_term_id). 

The Census MUST only include observations (cells) corresponding to the following values for [`assay_ontology_term_id`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#assay_ontology_term_id) if present in CELLxGENE Discover. 


| Accepted Assays |  |  |
|--------|--------|--------|
| [EFO:0003755](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0003755)<br>for <i>FL-cDNA</i> | [EFO:0008640](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008640)<br>for <i>3'T-fill</i> | [EFO:0008641](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008641)<br>for <i>3’-end-seq</i> |
| [EFO:0008643](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008643)<br>for <i>3′-Seq</i> | [EFO:0008661](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008661)<br>for <i>Bru-Seq</i> | [EFO:0008669](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008669)<br>for <i>CAGEscan</i> |
| [EFO:0008673](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008673)<br>for <i>CapSeq</i> | [EFO:0008675](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008675)<br>for <i>CaptureSeq</i> | [EFO:0008679](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008679)<br>for <i>CEL-seq</i> |
| [EFO:0008694](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008694)<br>for <i>ClickSeq</i> | [EFO:0008697](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008697)<br>for <i>cP-RNA-Seq</i> | [EFO:0008703](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008703)<br>for <i>CytoSeq</i> |
| [EFO:0008708](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008708)<br>for <i>DeepCAGE</i> | [EFO:0008710](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008710)<br>for <i>Digital RNA</i> | [EFO:0008718](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008718)<br>for <i>DP-Seq</i> |
| [EFO:0008720](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008720)<br>for <i>DroNc-seq</i> | [EFO:0008722](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008722)<br>for <i>Drop-seq</i> | [EFO:0008735](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008735)<br>for <i>FACS-seq</i> |
| [EFO:0008747](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008747)<br>for <i>FRISCR</i> | [EFO:0008748](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008748)<br>for <i>FRT-Seq</i> | [EFO:0008752](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008752)<br>for <i>GMUCT 1.0</i> |
| [EFO:0008753](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008753)<br>for <i>GMUCT 2.0</i> | [EFO:0008756](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008756)<br>for <i>GRO-CAP</i> | [EFO:0008763](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008763)<br>for <i>Hi-SCL</i> |
| [EFO:0008780](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008780)<br>for <i>inDrop</i> | [EFO:0008796](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008796)<br>for <i>MARS-seq</i> | [EFO:0008797](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008797)<br>for <i>MATQ-seq</i> |
| [EFO:0008824](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008824)<br>for <i>NanoCAGE</i> | [EFO:0008825](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008825)<br>for <i>Nanogrid RNA-Seq</i> | [EFO:0008826](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008826)<br>for <i>NET-Seq</i> |
| [EFO:0008850](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008850)<br>for <i>PAS-Seq</i> | [EFO:0008859](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008859)<br>for <i>PEAT</i> | [EFO:0008863](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008863)<br>for <i>PLATE-Seq</i> |
| [EFO:0008868](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008868)<br>for <i>PRO-cap</i> | [EFO:0008869](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008869)<br>for <i>PRO-seq</i> | [EFO:0008877](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008877)<br>for <i>Quartz-seq</i> |
| [EFO:0008896](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008896)<br>for <i>RNA-Seq</i> | [EFO:0008897](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008897)<br>for <i>RNAtag-Seq</i> | [EFO:0008898](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008898)<br>for <i>RNET-seq</i> |
| [EFO:0008903](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008903)<br>for <i>SC3-seq</i> | [EFO:0008919](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008919)<br>for <i>Seq-Well</i> | [EFO:0008929](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008929)<br>for <i>SMA</i> |
| [EFO:0008930](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008930)<br>for <i>Smart-seq</i> | [EFO:0008931](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008931)<br>for <i>Smart-seq2</i> | [EFO:0008937](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008937)<br>for <i>snDrop-seq</i> |
| [EFO:0008941](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008941)<br>for <i>sNuc-Seq</i> | [EFO:0008945](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008945)<br>for <i>SPET-seq</i> | [EFO:0008953](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008953)<br>for <i>STRT-seq</i> |
| [EFO:0008954](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008954)<br>for <i>STRT-seq-2i</i> | [EFO:0008956](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008956)<br>for <i>SUPeR-seq</i> | [EFO:0008962](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008962)<br>for <i>TARDIS</i> |
| [EFO:0008966](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008966)<br>for <i>TCR Chain Paring</i> | [EFO:0008967](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008967)<br>for <i>TCR-LA-MC PCR</i> | [EFO:0008972](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008972)<br>for <i>TL-seq</i> |
| [EFO:0008974](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008974)<br>for <i>Tomo-Seq</i> | [EFO:0008975](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008975)<br>for <i>TRAP-Seq</i> | [EFO:0008978](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008978)<br>for <i>TSS Sequencing</i> |
| [EFO:0008980](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008980)<br>for <i>UMI Method</i> | [EFO:0009309](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0009309)<br>for <i>Div-Seq</i> | [EFO:0009899](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0009899)<br>for <i>10x 3' v2</i> |
| [EFO:0009900](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0009900)<br>for <i>10x 5' v2</i> | [EFO:0009901](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0009901)<br>for <i>10x 3' v1</i> | [EFO:0009919](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0009919)<br>for <i>SPLiT-seq</i> |
| [EFO:0009922](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0009922)<br>for <i>10x 3' v3</i> | [EFO:0009991](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0009991)<br>for <i>Nuc-Seq</i> | [EFO:0009999](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0009999)<br>for <i>G&T-Seq</i> |
| [EFO:0010003](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010003)<br>for <i>RASL-seq</i> | [EFO:0010004](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010004)<br>for <i>SCRB-seq</i> | [EFO:0010005](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010005)<br>for <i>DR-Seq</i> |
| [EFO:0010006](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010006)<br>for <i>scM&T-seq</i> | [EFO:0010007](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010007)<br>for <i>scTrio-seq</i> | [EFO:0010010](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010010)<br>for <i>CEL-seq2</i> |
| [EFO:0010022](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010022)<br>for <i>Smart-3Seq</i> | [EFO:0010034](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010034)<br>for <i>Cappable-Seq</i> | [EFO:0010041](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010041)<br>for <i>Nascent-Seq</i> |
| [EFO:0010058](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010058)<br>for <i>Fluidigm C1-based SMARTer library preparation</i> | [EFO:0010184](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010184)<br>for <i>Smart-like</i> | [EFO:0010550](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010550)<br>for <i>sci-RNA-seq</i> |
| [EFO:0011025](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0011025)<br>for <i>10x 5' v1</i> | [EFO:0022396](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022396)<br>for <i>TruSeq</i> | [EFO:0022488](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022488)<br>for <i>Smart-seq3</i> |
| [EFO:0022490](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022490)<br>for <i>ScaleBio single cell RNA sequencing</i> | [EFO:0022600](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022600)<br>for <i>Parse Evercode Whole Transcriptome v1</i> | [EFO:0022601](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022601)<br>for <i>Parse Evercode Whole Transcriptome v2</i> |
| [EFO:0022602](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022602)<br>for <i>Parse Evercode Whole Transcriptome v3</i> | [EFO:0022604](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022604)<br>for <i>10x 3' v4</i> | [EFO:0022605](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022605)<br>for <i>10x 5' v3</i> |
| [EFO:0022606](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022606)<br>for <i>10x gene expression flex</i> | [EFO:0022839](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022839)<br>for <i>STORM-seq</i> | [EFO:0022845](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022845)<br>for <i>modified STRT-seq</i> |
| [EFO:0022846](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022846)<br>for <i>5' STRT-seq</i> | [EFO:0022857](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022857)<br>for <i>Visium Spatial Gene Expression V1</i>  | [EFO:0022962](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022962)<br>for <i>SHARE-seq</i> |
| [EFO:0030001](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030001)<br>for <i>sci-CAR</i> | [EFO:0030002](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030002)<br>for <i>microwell-seq</i> | [EFO:0030003](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030003)<br>for <i>10x 3' transcription profiling</i> |
| [EFO:0030004](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030004)<br>for <i>10x 5' transcription profiling</i> | [EFO:0030019](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030019)<br>for <i>Seq-Well S3</i> | [EFO:0030021](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030021)<br>for <i>Nx1-seq</i> |
| [EFO:0030026](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030026)<br>for <i>sci-Plex</i> | [EFO:0030028](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030028)<br>for <i>sci-RNA-seq3</i> | [EFO:0030030](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030030)<br>for <i>Quant-seq</i> |
| [EFO:0030031](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030031)<br>for <i>SCOPE-chip</i> | [EFO:0030059](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030059)<br>for <i>10x multiome</i> | [EFO:0030060](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030060)<br>for <i>mCT-seq</i> |
| [EFO:0030061](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030061)<br>for <i>mcSCRB-seq</i> | [EFO:0030062](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030062)<br>for <i>Slide-seqV2</i> | [EFO:0030074](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030074)<br>for <i>SORT-seq</i> |
| [EFO:0700003](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0700003)<br>for <i>BD Rhapsody Whole Transcriptome Analysis</i> | [EFO:0700004](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0700004)<br>for <i>BD Rhapsody Targeted mRNA</i> | [EFO:0700010](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0700010)<br>for <i>TruDrop</i> |
| [EFO:0700011](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0700011)<br>for <i>GEXSCOPE technology</i> | [EFO:0700016](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0700016)<br>for <i>Smart-seq v4</i> | [EFO:0900000](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0900000)<br>for <i>particle-templated instant partition sequencing</i> |
| [EFO:0900001](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0900001)<br>for <i>Asteria scRNA-seq kit</i> | [EFO:0900002](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0900002)<br>for <i>HIVE CLX Single-Cell RNAseq Solution</i> |  |

##### Spatial Assays

There are further requirements for two spatial assays:

* If [`assay_ontology_term_id`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#assay_ontology_term_id) is <a href="https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022857"><code>"EFO:0022857"</code></a> for <i>Visium Spatial Gene Expression V1</i>, then [<code>is_single</code>](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#is_single) MUST be <code>True</code> indicating that the dataset represents one Space Ranger output for a single tissue section.

* If [`assay_ontology_term_id`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#assay_ontology_term_id) is <a href="https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030062"><code>"EFO:0030062"</code></a> for <i>Slide-seqV2</i>, then [<code>is_single</code>](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#is_single) MUST be <code>True</code>, indicating that the dataset represents the output for a single array on a puck.

Otherwise, Census MUST NOT include the corresponding observations (cells).

##### Full-gene sequencing assays

Full-gene sequencing assays represent the subset of accepted assays that perform full-gene sequencing when used at the single-cell level. Such data MUST be normalized by gene length for downstream analysis.

| Full-gene sequencing assays |  |  |
|--------|--------|--------|
| [EFO:0003755](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0003755)<br>for <i>FL-cDNA</i> | [EFO:0008747](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008747)<br>for <i>FRISCR</i> | [EFO:0008763](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008763)<br>for <i>Hi-SCL</i> |
| [EFO:0008797](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008797)<br>for <i>MATQ-seq</i> | [EFO:0008877](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008877)<br>for <i>Quartz-seq</i> | [EFO:0008930](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008930)<br>for <i>Smart-seq</i> |
| [EFO:0008931](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008931)<br>for <i>Smart-seq2</i> | [EFO:0008956](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0008956)<br>for <i>SUPeR-seq</i> | [EFO:0009999](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0009999)<br>for <i>G&T-Seq</i> |
| [EFO:0010004](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010004)<br>for <i>SCRB-seq</i> | [EFO:0010006](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010006)<br>for <i>scM&T-seq</i> | [EFO:0010022](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010022)<br>for <i>Smart-3Seq</i> |
| [EFO:0010058](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010058)<br>for <i>Fluidigm C1-based<br>SMARTer library preparation</i> | [EFO:0010184](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0010184)<br>for <i>Smart-like</i> | [EFO:0022396](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022396)<br>for <i>TruSeq</i> |
| [EFO:0022488](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022488)<br>for <i>Smart-seq3</i> | [EFO:0022839](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0022839)<br>for <i>STORM-seq</i> | [EFO:0030031](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030031)<br>for <i>SCOPE-chip</i> |
| [EFO:0030060](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030060)<br>for <i>mCT-seq</i> | [EFO:0030061](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0030061)<br>for <i>mcSCRB-seq</i> | [EFO:0700016](https://www.ebi.ac.uk/ols4/ontologies/efo/classes?obo_id=EFO%3A0700016)<br>for <i>Smart-seq v4</i> |


#### Data matrix types

Per the CELLxGENE dataset schema, [all RNA assays MUST include UMI or read counts](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#x-matrix-layers). Author-normalized data layers [as defined in the CELLxGENE dataset schema](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#x-matrix-layers) MUST NOT be included in the Census.

#### Sample types

Observations (cells) with a [`tissue_type`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#tissue_type) value equal to "tissue" or "organoid" MUST be included in the Census. Observations with all other values of `tissue_type` such as "primary cell culture" MUST NOT be included.

#### Repeated data

When a cell is represented multiple times in CELLxGENE Discover, only one is marked as the primary cell. This is defined in the CELLxGENE dataset schema under [`is_primary_data`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#is_primary_data). This information MUST be included in the Census cell metadata to enable queries that retrieve datasets (see cell metadata below), and all cells MUST be included in the Census.

### Data encoding and organization

The Census MUST be encoded as a `SOMACollection` which is referenced as `census_obj` in the following sections. `census_obj` MUST have two keys `"census_info"` and `"census_data"` whose contents are defined in the sections below.

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
   2. value: Semver schema version.
2. Census build date:
   1. label: `"census_build_date"`
   2. value: The date this Census was built in ISO 8601 date format
3. Dataset schema version:
   1. label: `"dataset_schema_version"`
   2. value: The CELLxGENE Discover schema version of the source H5AD files.
4. Total number of cells or spatial spots included in this Census build:
   1. label: `"total_cell_count"`
   2. value: Cell count
5. Unique number of cells or spatial spots included in this Census build (is_primary_data == True)
   1. label: `"unique_cell_count"`
   2. value: Cell count
6. Number of human donors included in this Census build. Donors are guaranteed to be unique within datasets, not across all Census.
   1. label: `"number_donors_homo_sapiens"`
   2. value: Donor count
7. Number of mouse donors included in this Census build. Donors are guaranteed to be unique within datasets, not across all Census.
   1. label: `"number_donors_mus_musculus"`
   2. value: Donor count

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
    <td>2.0.0</td>
  </tr>
  <tr>
    <td>census_build_date</td>
    <td>2024-04-01</td>
  </tr>
  <tr>
    <td>dataset_schema_version </td>
    <td>5.1.0</td>
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
    <td>citation</td>
    <td>string</td>
    <td>As defined in the CELLxGENE schema.</td>
  </tr>
  <tr>
    <td>collection_id</td>
    <td>string</td>
    <td rowspan="6">As defined in CELLxGENE Discover <a href="https://api.cellxgene.cziscience.com/curation/ui/">data schema</a> (see &quot;Schemas&quot; section for field definitions)".</td>
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
    <td>collection_doi_label</td>
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
  <tr>
    <td>dataset_version_id</td>
    <td>string</td>
    <td>As defined in CELLxGENE Discover <a href="https://api.cellxgene.cziscience.com/curation/ui/">data schema</a> (see &quot;Schemas&quot; section for field definitions)".</td>
  </tr>
</tbody>
</table>

#### Census summary cell counts  – `census_obj["census_info"]["summary_cell_counts"]` – `SOMADataframe`

Summary cell counts grouped by organism and relevant cell metadata MUST be modeled as a `SOMADataFrame` in `census_obj["census_info"]["summary_cell_counts"]`. Each row MUST correspond to a combination of organism and metadata variables with the following columns:

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
    <td>The value of an <code>organism_label</code> defined in <a href="#census-table-of-organisms---census_objcensus_infoorganisms--somadataframe">Census table of organisms  – `census_obj["census_info"]["organisms"]` – `SOMADataframe`</a></td>
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
    <td>Label associated with an instance of metadata (e.g. <code>"lung"</code> if <code>category</code> is <code>"tissue"</code>). <code>"na"</code> if none.</td>
  </tr>
  <tr>
    <td>ontology_term_id</td>
    <td>string</td>
    <td>ID associated with an instance of metadata (e.g. <code>"UBERON:0002048"</code> if category is <code>"tissue"</code>). <code>"na"</code> if none.</td>
  </tr>
  <tr>
    <td>total_cell_count</td>
    <td>int</td>
    <td>Total number of cell counts for the combination of values of all other fields above.</td>
  </tr>
  <tr>
    <td>unique_cell_count</td>
    <td>int</td>
    <td>Unique number of cells for the combination of values of all other fields above. Unique number of cells refers to the cell count, for this group, when <code><a href="https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/5.1.0/schema.md#is_primary_data">is_primary_data == True</a></code> </td>
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
    <td>Homo sapiens</td>
    <td>all</td>
    <td>na</td>
    <td>na</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>cell_type</td>
    <td>cell_type_a</td>
    <td>CL:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>cell_type</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>cell_type</td>
    <td>cell_type_N</td>
    <td>CL:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>assay</td>
    <td>assay_a</td>
    <td>EFO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>assay</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>assay</td>
    <td>assay_N</td>
    <td>EFO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>tissue</td>
    <td>tissue_a</td>
    <td>UBERON:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>tissue</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>tissue</td>
    <td>tissue_N</td>
    <td>UBERON:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>tissue_general</td>
    <td>tissue_general_a</td>
    <td>UBERON:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>tissue_general</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>tissue_general</td>
    <td>tissue_general_N</td>
    <td>UBERON:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>disease</td>
    <td>disease_a</td>
    <td>MONDO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>disease</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>disease</td>
    <td>disease_N</td>
    <td>MONDO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>self_reported_ethnicity</td>
    <td>self_reported_ethnicity_a</td>
    <td>HANCESTRO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>self_reported_ethnicity</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>self_reported_ethnicity</td>
    <td>self_reported_ethnicity_N</td>
    <td>HANCESTRO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>sex</td>
    <td>sex_a</td>
    <td>PATO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>sex</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>sex</td>
    <td>sex_N</td>
    <td>PATO:XXXXX</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>suspension_type</td>
    <td>suspension_type_a</td>
    <td>na</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>suspension_type</td>
    <td>…</td>
    <td>…</td>
    <td>x</td>
    <td>x</td>
  </tr>
  <tr>
    <td>Homo sapiens</td>
    <td>suspension_type</td>
    <td>suspension_type_N</td>
    <td>na</td>
    <td>x</td>
    <td>x</td>
  </tr>
</tbody>
</table>

#### Census table of organisms  – `census_obj["census_info"]["organisms"]` – `SOMADataframe`

Information about organisms whose cells are included in the Census MUST be included in a table modeled as a `SOMADataFrame`. Each row MUST correspond to an individual organism with the following columns:

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
    <td>organism_ontology_term_id</td>
    <td>string</td>
    <td>As defined in the CELLxGENE dataset schema.</td>
  </tr>
  <tr>
    <td>organism_label</td>
    <td>string</td>
    <td>As defined in the CELLxGENE dataset schema.</td>
  </tr>
  <tr>
    <td>organism</td>
    <td>string</td>
    <td>Machine-friendly name for <a href="#census-data--census_objcensus_dataorganism--somaexperiment">Single Cell Census Data – `census_obj["census_data"][organism]` – `SOMAExperiment`</a>. Its value MUST be the result of:
    <ul>
      <li>Converting the <code>organism_label</code> to lowercase </li>
      <li>Replacing one or more consecutive spaces in the <code>organism_label</code> with a single underscore</li>
    </ul>
  </tr>
</tbody>
</table>

An example of this `SOMADataFrame`:

<table>
<thead>
  <tr>
    <th>organism_ontology_term_id</th>
    <th>organism_label</th>
    <th>organism</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>NCBITaxon:9483</td>
    <td>Callithrix jacchus</td>
    <td>callithrix_jacchus</td>
  </tr>
  <tr>
    <td>NCBITaxon:9606</td>
    <td>Homo sapiens</td>
    <td>homo_sapiens</td>
  </tr>
  <tr>
    <td>NCBITaxon:10090</td>
    <td>Mus musculus</td>
    <td>mus_musculus</td>
  </tr>
</tbody>
</table>

### Single Cell Census Data – `census_obj["census_data"][organism]` – `SOMAExperiment`

Non-spatial data for organisms MUST be stored as a `SOMAExperiment` in `census_obj["census_data"][organism]` where the value of <code>organism</code> matches an <code>organism</code> defined in <a href="#census-table-of-organisms---census_objcensus_infoorganisms--somadataframe">Census table of organisms  – `census_obj["census_info"]["organisms"]` – `SOMADataframe`</a>.

For example, non-spatial data for _Homo sapiens_ MUST be stored as a `SOMAExperiment` in `census_obj["census_data"]["homo_sapiens"]`.

For each organism with qualifying data, the `SOMAExperiment` MUST contain the following:

* Cell metadata – `census_obj["census_data"][organism].obs` – `SOMADataFrame`
* Data  –  `census_obj["census_data"][organism].ms` – `SOMACollection`. This `SOMACollection` MUST only contain one `SOMAMeasurement` in `census_obj["census_data"][organism].ms["RNA"]` with the following:
  * Matrix  data –  `census_obj["census_data"][organism].ms["RNA"].X` – `SOMACollection`. It MUST contain exactly one layer:
    * Count matrix – `census_obj["census_data"][organism].ms["RNA"].X["raw"]` – `SOMASparseNDArray`
    * Normalized count matrix – `census_obj["census_data"][organism].ms["RNA"].X["normalized"]` – `SOMASparseNDArray`
  * Feature metadata – `census_obj["census_data"][organism].ms["RNA"].var` – `SOMAIndexedDataFrame`
  * Feature dataset presence matrix – `census_obj["census_data"][organism].ms["RNA"]["feature_dataset_presence_matrix"]` – `SOMASparseNDArray`

#### Matrix Data, count (raw) matrix – `census_obj["census_data"][organism].ms["RNA"].X["raw"]` – `SOMASparseNDArray`

Per the CELLxGENE dataset schema, [all RNA assays MUST include UMI or read counts](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#x-matrix-layers). These counts MUST be encoded as `float32` in this `SOMASparseNDArray` with a fill value of zero (0), and no explicitly stored zero values.

#### Matrix Data, normalized count matrix – `census_obj["census_data"][organism].ms["RNA"].X["normalized"]` – `SOMASparseNDArray`

This is an experimental data artifact - it may be removed at any time.

A library-sized normalized layer, containing a normalized variant of the count (raw) matrix.
For [full-gene sequencing assays](#full-gene-sequencing-assays), given a value `X[i,j]` in the counts (raw) matrix, library-size normalized values are defined
as `normalized[i,j] = (X[i,j] / var[j].feature_length) / sum(X[i, ] / var.feature_length[j])`.
For all other assays, for a value `X[i,j]` in the counts (raw) matrix, library-size normalized values are defined
as `normalized[i,j] = X[i,j] / sum(X[i, ])`.

#### Feature metadata – `census_obj["census_data"][organism].ms["RNA"].var` – `SOMADataFrame`

The Census MUST only contain features with a [`feature_biotype`](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#feature_biotype) value of "gene".

The [gene references are pinned](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md#required-gene-annotations) as defined in the CELLxGENE dataset schema.

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
    <td>As defined in CELLxGENE dataset schema</td>
  </tr>
  <tr>
    <td>feature_length</td>
    <td>int</td>
    <td>As defined in CELLxGENE dataset schema</a>.</td>
  </tr>
  <tr>
    <td>nnz</td>
    <td>int64</td>
    <td>For this feature, the number of non-zero values present in the `X['raw']` counts (raw) matrix.</td>
  </tr>
  <tr>
    <td>n_measured_obs</td>
    <td>int64</td>
    <td>For this feature, the number of observations present in the source H5AD (sum of feature presence matrix).</td>
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
    <td>UBERON ontology term identifier for the <i>high-level tissue mapping</i> assigned by the <a href="https://github.com/chanzuckerberg/cellxgene-census/blob/main/tools/cellxgene_census_builder/src/cellxgene_census_builder/build_soma/tissue_mapper.py"><code>TissueMapper</code></a>.</td>
  </tr>
  <tr>
    <td>tissue_general</td>
    <td>string</td>
    <td>UBERON ontology label for the <i>high-level tissue mapping</i> assigned by the <a href="https://github.com/chanzuckerberg/cellxgene-census/blob/main/tools/cellxgene_census_builder/src/cellxgene_census_builder/build_soma/tissue_mapper.py"><code>TissueMapper</code></a>.</td>
  </tr>
  <tr>
    <td>assay_ontology_term_id</td>
    <td colspan="2" rowspan="19">As defined in CELLxGENE dataset schema</td>
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
    <td>observation_joinid</td>
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
  <tr>
    <td>tissue_type</td>
  </tr>
  <tr>
    <td>nnz</td>
    <td>int64</td>
    <td>For this observation, the number of non-zero measurements in the `X['raw']` counts (raw) matrix.</td>
  </tr>
  <tr>
    <td>n_measured_vars</td>
    <td>int64</td>
    <td>For this observation, the number of features present in the source H5AD (sum of feature presence matrix).</td>
  </tr>
  <tr>
    <td>raw_sum</td>
    <td>float32</td>
    <td>For this observation, the sum of the `X['raw']` counts (raw) matrix values.</td>
  </tr>
  <tr>
    <td>raw_mean_nnz</td>
    <td>float32</td>
    <td>For this observation, the mean of the `X['raw']` counts (raw) matrix values. Zeroes are excluded from the calculation.</td>
  </tr>
  <tr>
    <td>raw_variance_nnz</td>
    <td>float32</td>
    <td>For this observation, the variance of the `X['raw']` counts (raw) matrix values. Zeroes are excluded from the calculation.</td>
  </tr>
</tbody>
</table>

### Census Spatial Sequencing Data – `census_obj["census_spatial_sequencing"][organism]` – `SOMAExperiment`

Spatial data for organisms MUST be stored as a `SOMAExperiment` in `census_obj["census_data"][organism]` where the value of <code>organism</code> matches an <code>organism</code> defined in <a href="#census-table-of-organisms---census_objcensus_infoorganisms--somadataframe">Census table of organisms  – `census_obj["census_info"]["organisms"]` – `SOMADataframe`</a>.

For example, spatial data for _Homo sapiens_ MUST be stored as a `SOMAExperiment` in `census_obj["census_spatial_sequencing"]["homo_sapiens"]`.

**EDITORIAL NOTE: The assay references/restatement will be addressed when this section is rewritten**.

Only Visium Spatial Gene Expression V1 ("EFO:0022857") and Slide-seqV2 ("EFO:0030062") are supported for spatial data. [See the "assays included" section above](assays).

For each organism the `SOMAExperiment` MUST contain the following:

* Cell metadata – `census_obj["census_spatial_sequencing"][organism].obs` – `SOMADataFrame`
* Non-spatial data  –  `census_obj["census_spatial_sequencing"][organism].ms` – `SOMACollection`. This `SOMACollection` MUST only contain one `SOMAMeasurement` in `census_obj["census_spatial_sequencing"][organism].ms["RNA"]` with the following:
  * Matrix  data –  `census_obj["census_spatial_sequencing"][organism].ms["RNA"].X` – `SOMACollection`. It MUST contain exactly one layer:
    * Count matrix – `census_obj["census_spatial_sequencing"][organism].ms["RNA"].X["raw"]` – `SOMASparseNDArray`
  * Feature metadata – `census_obj["census_spatial_sequencing"][organism].ms["RNA"].var` – `SOMAIndexedDataFrame`
  * Feature dataset presence matrix – `census_obj["census_spatial_sequencing"][organism].ms["RNA"]["feature_dataset_presence_matrix"]` – `SOMASparseNDArray`
* Obs to spatial data mapping:
  * Obs to spatial data – `census_obj["census_spatial_sequencing"][organism].obs_spatial_presence` – `SOMADataFrame`
* Spatial data  –  `census_obj["census_spatial_sequencing"][organism].spatial` – `SOMACollection`.
  * Spatial Scenes with spatial data  –  `census_obj["census_spatial_sequencing"][organism].spatial[scene_id]`  – `SOMAScene`.  There will be as many Spatial Scenes as  spatial datasets. Each`SOMAScene` MUST contain the following:
    * Positions array – `census_obj["census_spatial_sequencing"][organism].spatial[scene_id].obsl["loc"]` – `SOMAPointCloudDataFrame`.
    * High resolution image  – `census_obj["census_spatial_sequencing"][organism].spatial[scene_id].img[library_id]["highres_image"]` – `MultiscaleImage`.

#### Matrix Data, count (raw) matrix – `census_obj["census_spatial_sequencing"][organism].ms["RNA"].X["raw"]` – `SOMASparseNDArray`

Spatial and non-spatial data share the [same requirements](#matrix-data-count-raw-matrix--census_objcensus_dataorganismmsrnaxraw--somasparsendarray).

#### Feature metadata – `census_obj["census_spatial_sequencing"][organism].ms["RNA"].var` – `SOMADataFrame`

Spatial and non-spatial data share the [same requirements](#feature-metadata--census_objcensus_dataorganismmsrnavar--somadataframe).

#### Feature dataset presence matrix – `census_obj["census_spatial_sequencing"][organism].ms["RNA"]["feature_dataset_presence_matrix"]` – `SOMASparseNDArray`

Spatial and non-spatial data share the [same requirements](#feature-dataset-presence-matrix--census_objcensus_dataorganismmsrnafeature_dataset_presence_matrix--somasparsendarray).

#### Cell metadata – `census_obj["census_spatial_sequencing"][organism].obs` – `SOMADataFrame`

Spatial and non-spatial data share the [same requirements](#cell-metadata--census_objcensus_dataorganismobs--somadataframe).

**Important note:** In addition, the following spatial `obs` columns from the CELLxGENE dataset schema MUST be included in this `SOMADataFrame`

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
    <td>array_col</td>
    <td colspan="2" rowspan="3">As defined in CELLxGENE dataset schema</td>
  </tr>
  <tr>
    <td>array_row</td>
  </tr>
  <tr>
    <td>in_tissue</td>
  </tr>
</tbody>
</table>

#### Obs to spatial mapping –  `census_obj["census_spatial_sequencing"][organism].obs_presence_matrix` – `SOMADataFrame`

It indicates the link between an observation and a scene.  Each row corresponds to an observation with the following columns:

<!-- markdownlint-disable reference-links-images -->
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
    <td>soma_joinid</td>
    <td>int</td>
    <td>It MUST be valid <code>soma_joinid</code> from <code>census_obj["census_spatial_sequencing"][organism].obs</code>.</td>
  </tr>
  </tr>
    <td>scene_id</td>
    <td>string</td>
    <td>It MUST be valid <code>scene_id</code> from <code>census_obj["census_spatial_sequencing"][organism].spatial</code>.</td>
  </tr>
  <tr>
    <td>data</td>
    <td>bool</td>
    <td>It MUST be <code>True</code> if the scene contains spatial information about the observation, otherwise it MUST be <code>False</code>.</td>
  </tr>
  </tbody>
</table>
<!-- markdownlint-enable reference-links-images -->

#### Positions array of a Scene – `census_obj["census_spatial_sequencing"][organism].spatial[scene_id].obsl["loc"]` – `SOMAPointCloudDataFrame`

`scene_id` MUST correspond to the values `scene_id` in `census_obj["census_spatial_sequencing"][organism].obs_presence_matrix`.

For each observation in each Scene, spatial array positions and additional positional metadata MUST be encoded as a `SOMAPointCloudDataFrame`.  Each row corresponds to an observation with the following columns:

<!-- markdownlint-disable reference-links-images -->
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
    <td>x</td>
    <td>float</td>
    <td>It MUST be the corresponding value in the <b>first</b> column of <code>obsm["spatial"]</code>. As defined in the CELLxGENE dataset schema.</td>
  </tr>
  </tr>
    <td>y</td>
    <td>float</td>
    <td>It MUST be the corresponding value in the <b>second</b> column of <code>obsm["spatial"]</code>. As defined in the CELLxGENE dataset schema.</td>
  </tr>
  <tr>
    <td>soma_joinid</td>
    <td>integer</td>
    <td>It MUST be valid <code>soma_joinid</code> from <code>census_obj["census_spatial_sequencing"][organism].obs</code>.</td>
  </tr>
  </tr>
  </tr>
 </tbody>
</table>
<!-- markdownlint-enable reference-links-images -->

If Visium Spatial Gene Expression V1 ("EFO:0022857"), the units for the spatial array positions are pixels from the full-resolution image.

The location dataframe MUST have the metadata fields:

<!-- markdownlint-disable reference-links-images -->
<table>
<thead>
  <tr>
    <th>Key</th>
    <th>Encoding</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>soma_geometry_type</td>
    <td>str</td>
    <td>MUST be "radius"</td>
  </tr>
  </tr>
    <td>soma_geometry</td>
    <td>float</td>
    <td>Radius of points: <code>diameter/2</code>. If Visium Spatial Gene Expression V1 ("EFO:0022857") <code>diameter</code> MUST be <code>.uns["spatial"][library_id]['spot_diameter_fullres']</code>, as defined in the CELLxGENE dataset schema. For Slide-seqV2 ("EFO:0030062") the radius is a small constant.</td>
  </tr>
 </tbody>
</table>
<!-- markdownlint-enable reference-links-images -->

#### Images of a Scene - `census_obj["census_spatial_sequencing"][organism].spatial[scene_soma_joinid].img[library_id]` – `Collection` of `MultiscaleImage`

Images of a Visium Spatial Gene Expression V1 ("EFO:0022857") scene MUST adhere to the following specifications. Other assays MUST NOT have images, and MUST NOT include the `img` collection.

`library_id` MUST be the corresponding value in the source H5AD slot `.uns["spatial"][library_id]`, as defined in the CELLxGENE dataset schema.

##### High resolution image of a Scene  – `census_obj["census_spatial_sequencing"][organism].spatial[scene_soma_joinid].img[library_id]["highres_image"]` – `MultiscaleImage`

The high resolution image of a Visium Spatial Gene Expression V1("EFO:0022857") scene MUST be included and MUST be encoded as a `SOMAImageNDArray`.

**Value:** the image from `uns["spatial"][library_id]['images']['hires']` as defined in the CELLxGENE dataset schema.

## Changelog

### Version 2.4.0

* Updated all CELLxGENE Discover dataset schema references from 5.2.0 to 7.0.0
* Updated all _Visium Spatial Gene Expression_ references to _Visium Spatial Gene Expression V1_
* Species
  * Renamed section from _Species_ to _Organisms_ for consistency
  * Added _Callithrix jacchus_
  * Added _Macaca mulatta_
  * Added _Pan troglodytes_
* Multi-species data constraints
  * Deleted section due to deprecated requirements for datasets containing multiple species or orthologous gene references
* Assays
  * Inlined and updated the Census Accepted Assays (CSV) file
    * Added <code>EFO:0008703</code> for <i>CytoSeq</i>  
    * Added <code>EFO:0009999</code> for <i>G&T-Seq</i>  
    * Added <code>EFO:0010005</code> for <i>DR-Seq</i>  
    * Added <code>EFO:0010006</code> for <i>scM&T-seq</i>  
    * Added <code>EFO:0010007</code> for <i>scTrio-seq</i>  
    * Added <code>EFO:0022604</code> for <i>10x 3' v4</i>  
    * Added <code>EFO:0022605</code> for <i>10x 5' v3</i>  
    * Added <code>EFO:0022857</code> for <i>Visium Spatial Gene Expression V1</i>  
    * Added <code>EFO:0022962</code> for <i>SHARE-seq</i>  
    * Added <code>EFO:0030001</code> for <i>sci-CAR</i>  
    * Added <code>EFO:0030026</code> for <i>sci-Plex</i>  
    * Added <code>EFO:0030059</code> for <i>10x multiome</i>  
    * Added <code>EFO:0030060</code> for <i>mCT-seq</i>  
    * Added <code>EFO:0900000</code> for <i>particle-templated instant partition sequencing</i>  
    * Added <code>EFO:0900001</code> for <i>Asteria scRNA-seq kit</i>  
    * Added <code>EFO:0900002</code> for <i>HIVE CLX Single-Cell RNAseq Solution</i>  
    * Removed <code>EFO:0008908</code> for <i>SCI-seq</i>
    * Removed <code>EFO:0010713</code> for <i>10x immune profiling</i>
    * Removed <code>EFO:0010714</code> for <i>10x TCR enrichment</i>
    * Removed <code>EFO:0010715</code> for <i>10x Ig enrichment</i>
    * Removed <code>EFO:0010961</code> for <i>Visium Spatial Gene Expression</i>
    * Removed <code>EFO:0010964</code> for <i>barcoded plate-based single cell RNA-seq</i>
    * Removed <code>EFO:0030078</code> for <i>droplet-based single-cell RNA library preparation</i>
  * Inlined and updated the Census Accepted Assays Full Gene (CSV) file
    * Added <code>EFO:0009999</code> for <i>G&T-Seq</i>
    * Added <code>EFO:0010006</code> for <i>scM&T-seq</i>
    * Added <code>EFO:0030060</code> for <i>mCT-seq</i>  
* Census table of organisms – `census_obj["census_info"]["organisms"]`
  * Replaced the code reference that documented the value of <code>organism</code> with its requirements
* Cell metadata – `census_obj["census_data"][organism].obs`
  * Corrected code references for <code>tissue_general_ontology_term_id</code> and <code>tissue_general</code>

### Version 2.3.0

* Added new top level object for holding spatial information, `census_spatial_sequencing`.

### Version 2.2.0

* Allow organoid data, i.e. `tissue_type` can now be `organoid` other than `tissue`.

### Version 2.1.0

* Update to require [CELLxGENE schema version 5.2.0](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/5.2.0/schema.md)
* Adds `collection_doi_label` to "Census table of CELLxGENE Discover datasets – `census_obj["census_info"]["datasets"]`"

### Version 2.0.1

* Update accepted assays for Census based on guidance from curators.

### Version 2.0.0

* Update to require [CELLxGENE schema version 5.0.0](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/5.0.0/schema.md)
* Expanded list of assays included in the Census.
* Expanded the list of assays defined as full-gene sequencing assays, which have special `normalized` layer handling.
* Clarified handling of datasets which are multi-species on the obs or var axis.

### Version 1.3.0

* Update to require [CELLxGENE schema version 4.0.0](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/4.0.0/schema.md)
* Adds `citation` to "Census table of CELLxGENE Discover datasets – `census_obj["census_info"]["datasets"]`"
* Adds `observation_joinid` and `tissue_type` to `obs` dataframe

### Version 1.2.0

* Update to require [CELLxGENE schema version 3.1.0](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.1.0/schema.md)

### Version 1.1.0

* Adds `dataset_version_id` to "Census table of CELLxGENE Discover datasets – `census_obj["census_info"]["datasets"]`"
* Add `X["normalized"]` layer
* Add `nnz` and `n_measured_obs` columns to `ms["RNA"].var` dataframe
* Add `nnz`, `n_measured_vars`, `raw_sum`, `raw_mean_nnz` and `raw_variance_nnz` columns to `obs` dataframe

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
