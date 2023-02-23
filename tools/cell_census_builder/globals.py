import time
from typing import Set

import pyarrow as pa
import tiledb
import tiledbsoma as soma

CENSUS_SCHEMA_VERSION = "0.1.1"

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

# top-level SOMA collection
CENSUS_INFO_NAME = "census_info"

# top-level SOMA collection
CENSUS_DATA_NAME = "census_data"

# "census_info"/"summary_cell_counts" SOMA Dataframe
CENSUS_SUMMARY_CELL_COUNTS_NAME = "summary_cell_counts"  # object name

# "census_info"/"summary_cell_counts" SOMA Dataframe
CENSUS_SUMMARY_NAME = "summary"

# "census_data"/{organism}/ms/"RNA" SOMA Matrix
MEASUREMENT_RNA_NAME = "RNA"

# "census_data"/{organism}/ms/"RNA"/"feature_dataset_presence_matrix" SOMA Matrix
FEATURE_DATASET_PRESENCE_MATRIX_NAME = "feature_dataset_presence_matrix"


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

"""
Materialization of obs/var schema in TileDB is tuned with the following, largely informed by empirical testing:
* string columns with repeating labels use DictionaryFilter, which efficiently encodes these highly repetative strings.
  Columns with non-repetative values (e.g., var.feature_id) are NOT dictionary coded.
* obs/var DataFrame show significant improvement in on-disk size and read performance at high Zstd compression level,
  making it worth the extra build/write compute time. Level set to highest value that does not require additional
  resources at decompression (nb. 20+ requres additional memory).
* int64 index columns (soma_joinid, soma_dim_0, etc) empirically show wins from ByteShuffle followed by Zstd.
  First dimension of axis dataframes are always monotonically increasing, and also beneit from DoubleDelta.
* Benchmarking X slicing (using lung demo notebook) used to tune X[raw]. Read / query performance did not benefit from
  higher Zstd compression beyond level=5, so the level was not increased further (and level=5 is still reasonable for
  writes)
"""

_RepetativeStringLabelObs = [
    # these columns are highly repetative string labels and will have appropriate filter
    "assay",
    "assay_ontology_term_id",
    "cell_type",
    "cell_type_ontology_term_id",
    "dataset_id",
    "development_stage",
    "development_stage_ontology_term_id",
    "disease",
    "disease_ontology_term_id",
    "donor_id",
    "self_reported_ethnicity",
    "self_reported_ethnicity_ontology_term_id",
    "sex",
    "sex_ontology_term_id",
    "suspension_type",
    "tissue",
    "tissue_ontology_term_id",
    "tissue_general",
    "tissue_general_ontology_term_id",
]
CENSUS_OBS_PLATFORM_CONFIG = {
    "tiledb": {
        "create": {
            "capacity": 2**16,
            "dims": {"soma_joinid": {"filters": ["DoubleDeltaFilter", {"_type": "ZstdFilter", "level": 19}]}},
            "attrs": {
                **{
                    k: {"filters": ["DictionaryFilter", {"_type": "ZstdFilter", "level": 19}]}
                    for k in _RepetativeStringLabelObs
                },
            },
            "offsets_filters": ["DoubleDeltaFilter", {"_type": "ZstdFilter", "level": 19}],
            "allows_duplicates": True,
        }
    }
}

CENSUS_VAR_TERM_COLUMNS = {
    "soma_joinid": pa.int64(),
    "feature_id": pa.large_string(),
    "feature_name": pa.large_string(),
    "feature_length": pa.int64(),
}
CENSUS_VAR_PLATFORM_CONFIG = {
    "tiledb": {
        "create": {
            "capacity": 2**16,
            "dims": {"soma_joinid": {"filters": ["DoubleDeltaFilter", {"_type": "ZstdFilter", "level": 19}]}},
            "offsets_filters": ["DoubleDeltaFilter", {"_type": "ZstdFilter", "level": 19}],
            "allows_duplicates": True,
        }
    }
}

CENSUS_X_LAYERS = {
    "raw": pa.float32(),
}
CENSUS_X_LAYERS_PLATFORM_CONFIG = {
    "raw": {
        "tiledb": {
            "create": {
                "capacity": 2**16,
                "dims": {
                    "soma_dim_0": {"tile": 2048, "filters": [{"_type": "ZstdFilter", "level": 5}]},
                    "soma_dim_1": {
                        "tile": 2048,
                        "filters": ["ByteShuffleFilter", {"_type": "ZstdFilter", "level": 5}],
                    },
                },
                "attrs": {"soma_data": {"filters": ["ByteShuffleFilter", {"_type": "ZstdFilter", "level": 5}]}},
                "cell_order": "row-major",
                "tile_order": "row-major",
                "allows_duplicates": True,
            },
        }
    }
}

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

# Feature_reference values which are ignored (not considered) in
# multi-organism filtering. Currently the null set.
FEATURE_REFERENCE_IGNORE: Set[str] = set()


"""
Singletons used throughout the package
"""

# Global SOMATileDBContext
_SOMA_TileDB_Context: soma.options.SOMATileDBContext = None

# Global TileDB context
_TileDB_Ctx: tiledb.Ctx = None

# The logical timestamp at which all builder data should be recorded
WRITE_TIMESTAMP = int(time.time() * 1000)

# Using "end of time" for read_timestamp means that all writes are visible, no matter what write timestamp was used
END_OF_TIME = 0xFFFFFFFFFFFFFFFF


def SOMA_TileDB_Context() -> soma.options.SOMATileDBContext:
    global _SOMA_TileDB_Context
    if _SOMA_TileDB_Context is None or _SOMA_TileDB_Context != TileDB_Ctx():
        # Set write timestamp to "now", so that we use consistent timestamps across all writes (mostly for aesthetic
        # reasons). Set read timestamps to be same as write timestamp so that post-build validation reads can "see"
        # the writes. Without setting read timestamp explicitly, the read timestamp would default to a time that
        # prevents seeing the builder's writes.
        _SOMA_TileDB_Context = soma.options.SOMATileDBContext(
            tiledb_ctx=TileDB_Ctx(),
            # TODO: Setting an explicit write timestamp causes later reads to fail!
            # write_timestamp=write_timestamp,
            # TODO: We *should* be able to set this equal to WRITE_TIMESTAMP, but as specifying a write_timestamp is
            #  problematic, we must use "end of time" for now
            read_timestamp=END_OF_TIME,
        )
    return _SOMA_TileDB_Context


def TileDB_Ctx() -> tiledb.Ctx:
    return _TileDB_Ctx


def set_tiledb_ctx(ctx: tiledb.Ctx) -> None:
    global _TileDB_Ctx, _SOMA_TileDB_Context
    _TileDB_Ctx = ctx
    _SOMA_TileDB_Context = None
