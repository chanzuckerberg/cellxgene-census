import functools
from typing import Set

import pyarrow as pa
import tiledb
import tiledbsoma as soma

from ..util import cpu_count

CENSUS_SCHEMA_VERSION = "1.2.0"

CXG_SCHEMA_VERSION = "4.0.0"  # the CELLxGENE schema version supported

# NOTE: The UBERON ontology URL needs to manually updated if the CXG Dataset Schema is updated. This is a temporary
# hassle, however, since the TissueMapper, which relies upon this ontology, will eventually be removed from the Builder
CXG_UBERON_ONTOLOGY_URL = "https://github.com/obophenotype/uberon/releases/download/v2023-06-28/uberon.owl"

# Columns expected in the census_datasets dataframe
CENSUS_DATASETS_COLUMNS = [
    "citation",
    "collection_id",
    "collection_name",
    "collection_doi",
    "dataset_id",
    "dataset_version_id",
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
    # Columns pulled from the CXG H5AD
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
    "observation_joinid": pa.large_string(),
    "self_reported_ethnicity": pa.large_string(),
    "self_reported_ethnicity_ontology_term_id": pa.large_string(),
    "sex": pa.large_string(),
    "sex_ontology_term_id": pa.large_string(),
    "suspension_type": pa.large_string(),
    "tissue": pa.large_string(),
    "tissue_ontology_term_id": pa.large_string(),
    "tissue_type": pa.large_string(),
}
CENSUS_OBS_STATS_COLUMNS = {
    # Columns computed during the Census build and written to the Census obs dataframe.
    "raw_sum": pa.float64(),
    "nnz": pa.int64(),
    "raw_mean_nnz": pa.float64(),
    "raw_variance_nnz": pa.float64(),
    "n_measured_vars": pa.int64(),
}
CENSUS_OBS_TERM_COLUMNS = {
    # Columns written to the Census obs dataframe.
    "soma_joinid": pa.int64(),
    "dataset_id": pa.large_string(),
    **CXG_OBS_TERM_COLUMNS,
    "tissue_general": pa.large_string(),
    "tissue_general_ontology_term_id": pa.large_string(),
    **CENSUS_OBS_STATS_COLUMNS,
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
    "tissue_type",
]
_NonRepeatitiveStringObs = [
    k
    for k in CENSUS_OBS_TERM_COLUMNS
    if (k not in _RepetativeStringLabelObs)
    and (pa.types.is_string(CENSUS_OBS_TERM_COLUMNS[k]) or pa.types.is_large_string(CENSUS_OBS_TERM_COLUMNS[k]))
]
_NumericObs = ["raw_sum", "nnz", "raw_mean_nnz", "raw_variance_nnz", "n_measured_vars"]
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
                **{k: {"filters": ["ByteShuffleFilter", {"_type": "ZstdFilter", "level": 9}]} for k in _NumericObs},
                **{k: {"filters": [{"_type": "ZstdFilter", "level": 19}]} for k in _NonRepeatitiveStringObs},
            },
            "offsets_filters": ["DoubleDeltaFilter", {"_type": "ZstdFilter", "level": 19}],
            "allows_duplicates": True,
        }
    }
}

CENSUS_VAR_STATS_COLUMNS = {
    # Columns computed during the Census build and written to the Census var dataframe.
    "nnz": pa.int64(),
    "n_measured_obs": pa.int64(),
}
CENSUS_VAR_TERM_COLUMNS = {
    # Columns written to the Census var dataframe.
    "soma_joinid": pa.int64(),
    "feature_id": pa.large_string(),
    "feature_name": pa.large_string(),
    "feature_length": pa.int64(),
    **CENSUS_VAR_STATS_COLUMNS,
}
_StringLabelVar = ["feature_id", "feature_name"]
_NumericVar = ["nnz", "n_measured_obs", "feature_length"]
CENSUS_VAR_PLATFORM_CONFIG = {
    "tiledb": {
        "create": {
            "capacity": 2**16,
            "dims": {
                "soma_joinid": {
                    "filters": [
                        "DoubleDeltaFilter",
                        {"_type": "ZstdFilter", "level": 19},
                    ]
                }
            },
            "attrs": {
                **{
                    k: {
                        "filters": [
                            {"_type": "ZstdFilter", "level": 19},
                        ]
                    }
                    for k in _StringLabelVar
                },
                **{
                    k: {
                        "filters": [
                            "ByteShuffleFilter",
                            {"_type": "ZstdFilter", "level": 9},
                        ]
                    }
                    for k in _NumericVar
                },
            },
            "offsets_filters": [
                "DoubleDeltaFilter",
                {"_type": "ZstdFilter", "level": 19},
            ],
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
                "soma_dim_0": {
                    "tile": 2048,
                    "filters": [{"_type": "ZstdFilter", "level": 5}],
                },
                "soma_dim_1": {
                    "tile": 2048,
                    "filters": [
                        "ByteShuffleFilter",
                        {"_type": "ZstdFilter", "level": 5},
                    ],
                },
            },
            "attrs": {
                "soma_data": {
                    "filters": [
                        "ByteShuffleFilter",
                        {"_type": "ZstdFilter", "level": 5},
                    ]
                }
            },
            "cell_order": "row-major",
            "tile_order": "row-major",
            "allows_duplicates": True,
        },
    }
}
CENSUS_X_LAYER_NORMALIZED_FLOAT_SCALE_FACTOR = 1.0 / 2**18
CENSUS_X_LAYERS_PLATFORM_CONFIG = {
    "raw": {
        **CENSUS_DEFAULT_X_LAYERS_PLATFORM_CONFIG,
    },
    "normalized": {
        "tiledb": {
            "create": {
                **CENSUS_DEFAULT_X_LAYERS_PLATFORM_CONFIG["tiledb"]["create"],
                "attrs": {
                    "soma_data": {
                        "filters": [
                            {
                                "_type": "FloatScaleFilter",
                                "factor": CENSUS_X_LAYER_NORMALIZED_FLOAT_SCALE_FACTOR,
                                "offset": 0.5,
                                "bytewidth": 4,
                            },
                            "ByteShuffleFilter",
                            {"_type": "ZstdFilter", "level": 5},
                        ]
                    }
                },
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
    "sm.compute_concurrency_level": min(cpu_count(), 64),
    "sm.io_concurrency_level": min(cpu_count(), 64),
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
