import pyarrow as pa
import tiledb

CENSUS_SCHEMA_VERSION = "0.0.1"

CXG_SCHEMA_VERSION = "3.0.0"  # version we write to the census
CXG_SCHEMA_VERSION_IMPORT = [CXG_SCHEMA_VERSION]  # versions we can ingest

# Columns expected in the census_datasets dataframe
CENSUS_DATASETS_COLUMNS = [
    "collection_id",
    "collection_name",
    "collection_doi",
    "dataset_id",
    "dataset_title",
    "dataset_h5ad_path",
    "dataset_total_cell_count",
]
CENSUS_DATASETS_NAME = "datasets"  # object name

CENSUS_SUMMARY_CELL_COUNTS_COLUMNS = {
    "organism": pa.string(),
    "category": pa.string(),
    "label": pa.string(),
    "ontology_term_id": pa.string(),
    "total_cell_count": pa.int64(),
    "unique_cell_count": pa.int64(),
}
CENSUS_SUMMARY_CELL_COUNTS_NAME = "summary_cell_counts"  # object name

CENSUS_SUMMARY_NAME = "summary"

# CXG schema columns we preserve in our census, and the Arrow type to encode as.  Schema:
# https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md
#
# NOTE: a few additional columns are added (they are not defined in the CXG schema),
# eg., dataset_id, tissue_general, etc.
CXG_OBS_TERM_COLUMNS = {
    "assay": pa.large_string(),
    "assay_ontology_term_id": pa.large_string(),
    "cell_type": pa.large_string(),
    "cell_type_ontology_term_id": pa.large_string(),
    "development_stage": pa.large_string(),
    "development_stage_ontology_term_id": pa.large_string(),
    "disease": pa.large_string(),
    "disease_ontology_term_id": pa.large_string(),
    "donor_id": pa.large_string(),
    "is_primary_data": pa.bool_(),
    "self_reported_ethnicity": pa.large_string(),
    "self_reported_ethnicity_ontology_term_id": pa.large_string(),
    "sex": pa.large_string(),
    "sex_ontology_term_id": pa.large_string(),
    "suspension_type": pa.large_string(),
    "tissue": pa.large_string(),
    "tissue_ontology_term_id": pa.large_string(),
}
CENSUS_OBS_TERM_COLUMNS = {
    "soma_joinid": pa.int64(),
    "dataset_id": pa.large_string(),
    **CXG_OBS_TERM_COLUMNS,
    "tissue_general": pa.large_string(),
    "tissue_general_ontology_term_id": pa.large_string(),
}

CENSUS_VAR_TERM_COLUMNS = {
    "soma_joinid": pa.int64(),
    "feature_id": pa.large_string(),
    "feature_name": pa.large_string(),
    "feature_length": pa.int64(),
}

X_LAYERS = [
    "raw",
]

# list of EFO terms that correspond to RNA seq modality/measurement
RNA_SEQ = [
    "EFO:0008720",  # DroNc-seq
    "EFO:0008722",  # Drop-seq
    "EFO:0008780",  # inDrop
    "EFO:0008913",  # single-cell RNA sequencing
    "EFO:0008919",  # Seq-Well
    "EFO:0008930",  # Smart-seq
    "EFO:0008931",  # Smart-seq2
    "EFO:0008953",  # STRT-seq
    "EFO:0008995",  # 10x technology
    "EFO:0009899",  # 10x 3' v2
    "EFO:0009900",  # 10x 5' v2
    "EFO:0009901",  # 10x 3' v1
    "EFO:0009922",  # 10x 3' v3
    "EFO:0010010",  # CEL-seq2
    "EFO:0010183",  # single cell library construction
    "EFO:0010550",  # sci-RNA-seq
    "EFO:0011025",  # 10x 5' v1
    "EFO:0030002",  # microwell-seq
    "EFO:0030003",  # 10x 3' transcription profiling
    "EFO:0030004",  # 10x 5' transcription profiling
    "EFO:0030019",  # Seq-Well S3
    "EFO:0700003",  # BD Rhapsody Whole Transcriptome Analysis
    "EFO:0700004",  # BD Rhapsody Targeted mRNA
]

DONOR_ID_IGNORE = ["pooled", "unknown"]

# Feature_reference values which are ignored (not considered) for
# multi-organism filtering.
SARS_COV_2 = "NCBITaxon:2697049"
ERCC_SPIKE_INS = "NCBITaxon:32630"
FEATURE_REFERENCE_IGNORE = {SARS_COV_2, ERCC_SPIKE_INS}


"""
Singletons used throughout the package
"""

# Global TileDB context
_TileDB_Ctx: tiledb.Ctx = None


def TileDB_Ctx() -> tiledb.Ctx:
    return _TileDB_Ctx


def set_tiledb_ctx(ctx: tiledb.Ctx) -> None:
    global _TileDB_Ctx
    _TileDB_Ctx = ctx
