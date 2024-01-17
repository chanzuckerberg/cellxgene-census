import functools
from typing import Any, List, Set, Tuple, Union

import pyarrow as pa
import tiledb
import tiledbsoma as soma

from ..util import cpu_count
from .schema_util import FieldSpec, TableSpec

# Feature flag - enables/disables use of Arrow dictionary / TileDB enum for
# DataFrame columns. True is enabled, False is disabled. Usage currently blocked
# by several TileDB-SOMA bugs.
USE_ARROW_DICTIONARY = False

CENSUS_SCHEMA_VERSION = "1.3.0"

CXG_SCHEMA_VERSION = "4.0.0"  # the CELLxGENE schema version supported

# NOTE: The UBERON ontology URL needs to manually updated if the CXG Dataset Schema is updated. This is a temporary
# hassle, however, since the TissueMapper, which relies upon this ontology, will eventually be removed from the Builder
CXG_UBERON_ONTOLOGY_URL = "https://github.com/obophenotype/uberon/releases/download/v2023-06-28/uberon.owl"

# Columns expected in the census_datasets dataframe
CENSUS_DATASETS_TABLE_SPEC = TableSpec.create(
    [
        ("soma_joinid", pa.int64()),
        ("citation", pa.large_string()),
        ("collection_id", pa.large_string()),
        ("collection_name", pa.large_string()),
        ("collection_doi", pa.large_string()),
        ("dataset_id", pa.large_string()),
        ("dataset_version_id", pa.large_string()),
        ("dataset_title", pa.large_string()),
        ("dataset_h5ad_path", pa.large_string()),
        ("dataset_total_cell_count", pa.int64()),
    ],
    use_arrow_dictionary=USE_ARROW_DICTIONARY,
)
CENSUS_DATASETS_NAME = "datasets"  # object name

CENSUS_SUMMARY_CELL_COUNTS_TABLE_SPEC = TableSpec.create(
    [
        ("soma_joinid", pa.int64()),
        FieldSpec(name="organism", type=pa.string(), is_dictionary=True),
        FieldSpec(name="category", type=pa.string(), is_dictionary=True),
        ("label", pa.string()),
        ("ontology_term_id", pa.string()),
        ("total_cell_count", pa.int64()),
        ("unique_cell_count", pa.int64()),
    ],
    use_arrow_dictionary=USE_ARROW_DICTIONARY,
)

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
# https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/4.0.0/schema.md
#
# NOTE: a few additional columns are added (they are not defined in the CXG schema),
# eg., dataset_id, tissue_general, etc.
#
CXG_OBS_TERM_COLUMNS = [  # Columns pulled from the CXG H5AD without modification.
    "assay",
    "assay_ontology_term_id",
    "cell_type",
    "cell_type_ontology_term_id",
    "development_stage",
    "development_stage_ontology_term_id",
    "disease",
    "disease_ontology_term_id",
    "donor_id",
    "is_primary_data",
    "observation_joinid",
    "self_reported_ethnicity",
    "self_reported_ethnicity_ontology_term_id",
    "sex",
    "sex_ontology_term_id",
    "suspension_type",
    "tissue",
    "tissue_ontology_term_id",
    "tissue_type",
]
CXG_OBS_COLUMNS_READ: Tuple[str, ...] = (  # Columns READ from the CXG H5AD - see open_anndata()
    *CXG_OBS_TERM_COLUMNS,
    "organism",
    "organism_ontology_term_id",
)
CENSUS_OBS_STATS_COLUMNS = ["raw_sum", "nnz", "raw_mean_nnz", "raw_variance_nnz", "n_measured_vars"]
CENSUS_OBS_FIELDS: List[Union[FieldSpec, Tuple[str, pa.DataType]]] = [
    ("soma_joinid", pa.int64()),
    FieldSpec(name="dataset_id", type=pa.string(), is_dictionary=True),
    FieldSpec(name="assay", type=pa.string(), is_dictionary=True),
    FieldSpec(name="assay_ontology_term_id", type=pa.string(), is_dictionary=True),
    FieldSpec(name="cell_type", type=pa.string(), is_dictionary=True),
    FieldSpec(name="cell_type_ontology_term_id", type=pa.string(), is_dictionary=True),
    FieldSpec(name="development_stage", type=pa.string(), is_dictionary=True),
    FieldSpec(name="development_stage_ontology_term_id", type=pa.string(), is_dictionary=True),
    FieldSpec(name="disease", type=pa.string(), is_dictionary=True),
    FieldSpec(name="disease_ontology_term_id", type=pa.string(), is_dictionary=True),
    FieldSpec(name="donor_id", type=pa.string(), is_dictionary=True),
    ("is_primary_data", pa.bool_()),
    ("observation_joinid", pa.large_string()),
    FieldSpec(name="self_reported_ethnicity", type=pa.string(), is_dictionary=True),
    FieldSpec(name="self_reported_ethnicity_ontology_term_id", type=pa.string(), is_dictionary=True),
    FieldSpec(name="sex", type=pa.string(), is_dictionary=True),
    FieldSpec(name="sex_ontology_term_id", type=pa.string(), is_dictionary=True),
    FieldSpec(name="suspension_type", type=pa.string(), is_dictionary=True),
    FieldSpec(name="tissue", type=pa.string(), is_dictionary=True),
    FieldSpec(name="tissue_ontology_term_id", type=pa.string(), is_dictionary=True),
    FieldSpec(name="tissue_type", type=pa.string(), is_dictionary=True),
    FieldSpec(name="tissue_general", type=pa.string(), is_dictionary=True),
    FieldSpec(name="tissue_general_ontology_term_id", type=pa.string(), is_dictionary=True),
    ("raw_sum", pa.float64()),
    ("nnz", pa.int64()),
    ("raw_mean_nnz", pa.float64()),
    ("raw_variance_nnz", pa.float64()),
    ("n_measured_vars", pa.int64()),
]
CENSUS_OBS_TABLE_SPEC = TableSpec.create(CENSUS_OBS_FIELDS, use_arrow_dictionary=USE_ARROW_DICTIONARY)

"""
Materialization (filter pipelines, capacity, etc) of obs/var schema in TileDB is tuned by empirical testing.
"""
# Numeric columns
_NumericObsAttrs = ["raw_sum", "nnz", "raw_mean_nnz", "raw_variance_nnz", "n_measured_vars"]
# Categorical/dict-like columns
_DictLikeObsAttrs = [
    f.name
    for f in CENSUS_OBS_FIELDS
    if isinstance(f, FieldSpec) and f.is_dictionary
    if f.is_dictionary and f.name not in (_NumericObsAttrs + ["soma_joinid"])
]
# Best of the rest
_AllOtherObsAttrs = [
    f.name
    for f in CENSUS_OBS_TABLE_SPEC.fields
    if f.name not in (_DictLikeObsAttrs + _NumericObsAttrs + ["soma_joinid"])
]
# Dict filter varies depending on whether we are using dictionary types in the schema
_DictLikeFilter: List[Any] = (
    [{"_type": "ZstdFilter", "level": 19}]
    if USE_ARROW_DICTIONARY
    else ["DictionaryFilter", {"_type": "ZstdFilter", "level": 19}]
)
CENSUS_OBS_PLATFORM_CONFIG = {
    "tiledb": {
        "create": {
            "capacity": 2**16,
            "dims": {"soma_joinid": {"filters": ["DoubleDeltaFilter", {"_type": "ZstdFilter", "level": 19}]}},
            "attrs": {
                **{
                    k: {"filters": ["ByteShuffleFilter", {"_type": "ZstdFilter", "level": 9}]} for k in _NumericObsAttrs
                },
                **{k: {"filters": _DictLikeFilter} for k in _DictLikeObsAttrs},
                **{k: {"filters": [{"_type": "ZstdFilter", "level": 19}]} for k in _AllOtherObsAttrs},
            },
            "offsets_filters": ["DoubleDeltaFilter", {"_type": "ZstdFilter", "level": 19}],
            "allows_duplicates": True,
        }
    }
}

CXG_VAR_COLUMNS_READ: Tuple[str, ...] = (
    "_index",
    "feature_name",
    "feature_length",
    "feature_reference",
    "feature_biotype",
)
CENSUS_VAR_TABLE_SPEC = TableSpec.create(
    [
        ("soma_joinid", pa.int64()),
        ("feature_id", pa.large_string()),
        ("feature_name", pa.large_string()),
        ("feature_length", pa.int64()),
        ("nnz", pa.int64()),
        ("n_measured_obs", pa.int64()),
    ],
    use_arrow_dictionary=USE_ARROW_DICTIONARY,
)
_StringLabelVar = ["feature_id", "feature_name"]
_NumericVar = ["nnz", "n_measured_obs", "feature_length"]
CENSUS_VAR_PLATFORM_CONFIG = {
    "tiledb": {
        "create": {
            "capacity": 2**16,
            "dims": {"soma_joinid": {"filters": ["DoubleDeltaFilter", {"_type": "ZstdFilter", "level": 19}]}},
            "attrs": {
                **{k: {"filters": [{"_type": "ZstdFilter", "level": 19}]} for k in _StringLabelVar},
                **{k: {"filters": ["ByteShuffleFilter", {"_type": "ZstdFilter", "level": 9}]} for k in _NumericVar},
            },
            "offsets_filters": ["DoubleDeltaFilter", {"_type": "ZstdFilter", "level": 19}],
            "allows_duplicates": True,
        }
    }
}

CENSUS_X_LAYERS = {
    "raw": pa.float32(),
    "normalized": pa.float32(),
}
CENSUS_DEFAULT_X_LAYERS_PLATFORM_CONFIG = {
    "tiledb": {
        "create": {
            "capacity": 2**16,
            "dims": {
                "soma_dim_0": {"tile": 2048, "filters": [{"_type": "ZstdFilter", "level": 9}]},
                "soma_dim_1": {"tile": 2048, "filters": ["ByteShuffleFilter", {"_type": "ZstdFilter", "level": 9}]},
            },
            "attrs": {"soma_data": {"filters": ["ByteShuffleFilter", {"_type": "ZstdFilter", "level": 9}]}},
            "cell_order": "row-major",
            "tile_order": "row-major",
            "allows_duplicates": True,
        },
    }
}
CENSUS_X_LAYERS_PLATFORM_CONFIG = {
    "raw": {
        **CENSUS_DEFAULT_X_LAYERS_PLATFORM_CONFIG,
    },
    "normalized": {
        "tiledb": {
            "create": {
                **CENSUS_DEFAULT_X_LAYERS_PLATFORM_CONFIG["tiledb"]["create"],
                "attrs": {"soma_data": {"filters": [{"_type": "ZstdFilter", "level": 19}]}},
            }
        }
    },
}

# list of EFO terms that correspond to RNA seq modality/measurement. These terms
# define the inclusive filter applied to obs.assay_ontology_term_id. All other
# terms are excluded from the Census.
RNA_SEQ = [
    "EFO:0008720",  # DroNc-seq
    "EFO:0008722",  # Drop-seq
    "EFO:0008780",  # inDrop
    "EFO:0008796",  # MARS-seq
    "EFO:0008919",  # Seq-Well
    "EFO:0008930",  # Smart-seq
    "EFO:0008931",  # Smart-seq2
    "EFO:0008953",  # STRT-seq
    "EFO:0009899",  # 10x 3' v2
    "EFO:0009900",  # 10x 5' v2
    "EFO:0009901",  # 10x 3' v1
    "EFO:0009922",  # 10x 3' v3
    "EFO:0010010",  # CEL-seq2
    "EFO:0010550",  # sci-RNA-seq
    "EFO:0011025",  # 10x 5' v1
    "EFO:0030002",  # microwell-seq
    "EFO:0030003",  # 10x 3' transcription profiling
    "EFO:0030004",  # 10x 5' transcription profiling
    "EFO:0030019",  # Seq-Well S3
    "EFO:0700003",  # BD Rhapsody Whole Transcriptome Analysis
    "EFO:0700004",  # BD Rhapsody Targeted mRNA
    "EFO:0700010",  # TruDrop
    "EFO:0700011",  # GEXSCOPE technology
    "EFO:0700016",  # Smart-seq v4
]

# Smart-Seq has special handling in the "normalized" X layers
SMART_SEQ = [
    "EFO:0008930",  # Smart-seq
    "EFO:0008931",  # Smart-seq2
    "EFO:0700016",  # Smart-seq v4
]

DONOR_ID_IGNORE = ["pooled", "unknown"]

# Feature_reference values which are ignored (not considered) in
# multi-organism filtering. Currently the null set.
FEATURE_REFERENCE_IGNORE: Set[str] = set()


# The default configuration for TileDB contexts used in the builder.
# Ref: https://docs.tiledb.com/main/how-to/configuration#configuration-parameters
DEFAULT_TILEDB_CONFIG = {
    "py.init_buffer_bytes": 1 * 1024**3,
    "py.deduplicate": "true",
    "soma.init_buffer_bytes": 1 * 1024**3,
    "sm.mem.reader.sparse_global_order.ratio_array_data": 0.3,
    #
    # Concurrency levels are capped for high-CPU boxes. Left unchecked, some
    # of the largest host configs can bump into Linux kernel thread limits,
    # without any real benefit to overall performance.
    "sm.compute_concurrency_level": min(cpu_count(), 128),
    "sm.io_concurrency_level": min(cpu_count(), 128),
}


"""
Singletons used throughout the package
"""


@functools.cache
def SOMA_TileDB_Context() -> soma.options.SOMATileDBContext:
    return soma.options.SOMATileDBContext(tiledb_ctx=TileDB_Ctx(), timestamp=None)


@functools.cache
def TileDB_Ctx() -> tiledb.Ctx:
    return tiledb.Ctx(DEFAULT_TILEDB_CONFIG)
