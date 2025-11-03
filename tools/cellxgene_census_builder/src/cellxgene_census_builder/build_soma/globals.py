import functools

import pyarrow as pa
import tiledbsoma as soma

from ..util import cpu_count
from .schema_util import FieldSpec, TableSpec

# Feature flag - enables/disables use of Arrow dictionary / TileDB enum for
# DataFrame columns. True is enabled, False is disabled.
USE_ARROW_DICTIONARY = True

CENSUS_SCHEMA_VERSION = "2.4.0"

CXG_SCHEMA_VERSION = "7.0.0"  # the CELLxGENE schema version supported

# Columns expected in the census_datasets dataframe
CENSUS_DATASETS_TABLE_SPEC = TableSpec.create(
    [
        ("soma_joinid", pa.int64()),
        ("citation", pa.large_string()),
        ("collection_id", pa.large_string()),
        ("collection_name", pa.large_string()),
        ("collection_doi", pa.large_string()),
        ("collection_doi_label", pa.large_string()),
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

# top-level SOMA collection
CENSUS_SPATIAL_SEQUENCING_NAME = "census_spatial_sequencing"

# "census_info"/"summary_cell_counts" SOMA Dataframe
CENSUS_SUMMARY_CELL_COUNTS_NAME = "summary_cell_counts"  # object name

# "census_info"/"summary_cell_counts" SOMA Dataframe
CENSUS_SUMMARY_NAME = "summary"

# "census_info"/"organisms" SOMA Dataframe
CENSUS_INFO_ORGANISMS_NAME = "organisms"

# "census_data"/{organism}/ms/"RNA" SOMA Matrix
MEASUREMENT_RNA_NAME = "RNA"

# "census_data"/{organism}/ms/"RNA"/"feature_dataset_presence_matrix" SOMA Matrix
FEATURE_DATASET_PRESENCE_MATRIX_NAME = "feature_dataset_presence_matrix"


# CXG schema columns we preserve in our census, and the Arrow type to encode as.  Schema:
# https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/5.0.0/schema.md
#
# NOTE: a few additional columns are added (they are not defined in the CXG schema),
# eg., dataset_id, tissue_general, etc.
#
# IMPORTANT: for any `obs` column, use Arrow `large_string` and `large_binary`, rather
# than `string` or `binary`. There is no at-rest difference (TileDB-SOMA encodes both as large),
# but the in-memory Arrow Array indices for string/binary can overflow as cell counts increase.
#
CXG_OBS_TERM_FIELDS = [  # Columns pulled from the CXG H5AD without modification.
    FieldSpec(name="assay", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="assay_ontology_term_id", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="cell_type", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="cell_type_ontology_term_id", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="development_stage", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="development_stage_ontology_term_id", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="disease", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="disease_ontology_term_id", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="donor_id", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="is_primary_data", type=pa.bool_(), is_dictionary=False),
    FieldSpec(name="observation_joinid", type=pa.large_string(), is_dictionary=False),
    FieldSpec(name="self_reported_ethnicity", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="self_reported_ethnicity_ontology_term_id", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="sex", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="sex_ontology_term_id", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="suspension_type", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="tissue", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="tissue_ontology_term_id", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="tissue_type", type=pa.large_string(), is_dictionary=True),
]
CXG_OBS_TERM_FIELDS_SPATIAL = [  # Spatial speicific columns
    FieldSpec(name="in_tissue", type=pa.int64(), is_dictionary=False),
    FieldSpec(name="array_row", type=pa.int64(), is_dictionary=False),
    FieldSpec(name="array_col", type=pa.int64(), is_dictionary=False),
]
CENSUS_OBS_FIELDS_MAPPED = [  # Annotation columns created by census builder
    FieldSpec(name="tissue_general", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="tissue_general_ontology_term_id", type=pa.large_string(), is_dictionary=True),
]
CENSUS_OBS_STATS_FIELDS = [  # Columns for stats calculated during build
    FieldSpec(name="raw_sum", type=pa.float64(), is_dictionary=False),
    FieldSpec(name="nnz", type=pa.int64(), is_dictionary=False),
    FieldSpec(name="raw_mean_nnz", type=pa.float64(), is_dictionary=False),
    FieldSpec(name="raw_variance_nnz", type=pa.float64(), is_dictionary=False),
    FieldSpec(name="n_measured_vars", type=pa.int64(), is_dictionary=False),
]
CENSUS_OBS_CORE_FIELDS = [
    FieldSpec(name="soma_joinid", type=pa.int64(), is_dictionary=False),
    FieldSpec(name="dataset_id", type=pa.large_string(), is_dictionary=True),
]
# These are not actually written, so pyarrow type is not important
CXG_OBS_FIELDS_READ = [  # Columns READ from the CXG H5AD - see open_anndata()
    FieldSpec(name="organism", type=pa.large_string(), is_dictionary=True),
    FieldSpec(name="organism_ontology_term_id", type=pa.large_string(), is_dictionary=True),
]

CXG_VAR_COLUMNS_READ: tuple[str, ...] = (
    "_index",
    "feature_name",
    "feature_length",
    "feature_reference",
    "feature_biotype",
    "feature_type",
)
CENSUS_VAR_TABLE_SPEC = TableSpec.create(
    [
        ("soma_joinid", pa.int64()),
        ("feature_id", pa.large_string()),
        ("feature_name", pa.large_string()),
        ("feature_type", pa.large_string()),
        ("feature_length", pa.int64()),
        ("nnz", pa.int64()),
        ("n_measured_obs", pa.int64()),
    ],
    use_arrow_dictionary=USE_ARROW_DICTIONARY,
)
_StringLabelVar = ["feature_id", "feature_name", "feature_type"]
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
                "attrs": {"soma_data": {"filters": [{"_type": "ZstdFilter", "level": 13}]}},
            }
        }
    },
}
CENSUS_POINT_CLOUD_PLATFORM_CONFIG = {
    "tiledb": {
        "create": {
            "allows_duplicates": True,
        }
    }
}

# list of EFO terms that correspond to RNA seq modality/measurement. These terms
# define the inclusive filter applied to obs.assay_ontology_term_id. All other
# terms are excluded from the Census.
RNA_SEQ = [
    "EFO:0003755",  # FL-cDNA
    "EFO:0008640",  # 3'T-fill
    "EFO:0008641",  # 3’-end-seq
    "EFO:0008643",  # 3′-Seq
    "EFO:0008661",  # Bru-Seq
    "EFO:0008669",  # CAGEscan
    "EFO:0008673",  # CapSeq
    "EFO:0008675",  # CaptureSeq
    "EFO:0008679",  # CEL-seq
    "EFO:0008694",  # ClickSeq
    "EFO:0008697",  # cP-RNA-Seq
    "EFO:0008703",  # CytoSeq
    "EFO:0008708",  # DeepCAGE
    "EFO:0008710",  # Digital RNA
    "EFO:0008718",  # DP-Seq
    "EFO:0008720",  # DroNc-seq
    "EFO:0008722",  # Drop-seq
    "EFO:0008735",  # FACS-seq
    "EFO:0008747",  # FRISCR
    "EFO:0008748",  # FRT-Seq
    "EFO:0008752",  # GMUCT 1.0
    "EFO:0008753",  # GMUCT 2.0
    "EFO:0008756",  # GRO-CAP
    "EFO:0008763",  # Hi-SCL
    "EFO:0008780",  # inDrop
    "EFO:0008796",  # MARS-seq
    "EFO:0008797",  # MATQ-seq
    "EFO:0008824",  # NanoCAGE
    "EFO:0008825",  # Nanogrid RNA-Seq
    "EFO:0008826",  # NET-Seq
    "EFO:0008850",  # PAS-Seq
    "EFO:0008859",  # PEAT
    "EFO:0008863",  # PLATE-Seq
    "EFO:0008868",  # PRO-cap
    "EFO:0008869",  # PRO-seq
    "EFO:0008877",  # Quartz-seq
    "EFO:0008896",  # RNA-Seq
    "EFO:0008897",  # RNAtag-Seq
    "EFO:0008898",  # RNET-seq
    "EFO:0008903",  # SC3-seq
    "EFO:0008919",  # Seq-Well
    "EFO:0008929",  # SMA
    "EFO:0008930",  # Smart-seq
    "EFO:0008931",  # Smart-seq2
    "EFO:0008937",  # snDrop-seq
    "EFO:0008941",  # sNuc-Seq
    "EFO:0008945",  # SPET-seq
    "EFO:0008953",  # STRT-seq
    "EFO:0008954",  # STRT-seq-2i
    "EFO:0008956",  # SUPeR-seq
    "EFO:0008962",  # TARDIS
    "EFO:0008966",  # TCR Chain Paring
    "EFO:0008967",  # TCR-LA-MC PCR
    "EFO:0008972",  # TL-seq
    "EFO:0008974",  # Tomo-Seq
    "EFO:0008975",  # TRAP-Seq
    "EFO:0008978",  # TSS Sequencing
    "EFO:0008980",  # UMI Method
    "EFO:0009309",  # Div-Seq
    "EFO:0009899",  # 10x 3' v2
    "EFO:0009900",  # 10x 5' v2
    "EFO:0009901",  # 10x 3' v1
    "EFO:0009919",  # SPLiT-seq
    "EFO:0009922",  # 10x 3' v3
    "EFO:0009991",  # Nuc-Seq
    "EFO:0009999",  # G&T-Seq
    "EFO:0010003",  # RASL-seq
    "EFO:0010004",  # SCRB-seq
    "EFO:0010005",  # DR-Seq
    "EFO:0010006",  # scM&T-seq
    "EFO:0010007",  # scTrio-seq
    "EFO:0010010",  # CEL-seq2
    "EFO:0010022",  # Smart-3Seq
    "EFO:0010034",  # Cappable-Seq
    "EFO:0010041",  # Nascent-Seq
    "EFO:0010058",  # Fluidigm C1-based SMARTer library preparation
    "EFO:0010184",  # Smart-like
    "EFO:0010550",  # sci-RNA-seq
    "EFO:0011025",  # 10x 5' v1
    "EFO:0022396",  # TruSeq
    "EFO:0022488",  # Smart-seq3
    "EFO:0022490",  # ScaleBio single cell RNA sequencing
    "EFO:0022600",  # Parse Evercode Whole Transcriptome v1
    "EFO:0022601",  # Parse Evercode Whole Transcriptome v2
    "EFO:0022602",  # Parse Evercode Whole Transcriptome v3
    "EFO:0022604",  # 10x 3' v4
    "EFO:0022605",  # 10x 5' v3
    "EFO:0022606",  # 10x gene expression flex
    "EFO:0022839",  # STORM-seq
    "EFO:0022845",  # modified STRT-seq
    "EFO:0022846",  # 5' STRT-seq
    "EFO:0022962",  # SHARE-seq
    "EFO:0030001",  # sci-CAR
    "EFO:0030002",  # microwell-seq
    "EFO:0030003",  # 10x 3' transcription profiling
    "EFO:0030004",  # 10x 5' transcription profiling
    "EFO:0030019",  # Seq-Well S3
    "EFO:0030021",  # Nx1-seq
    "EFO:0030026",  # sci-Plex
    "EFO:0030028",  # sci-RNA-seq3
    "EFO:0030030",  # Quant-seq
    "EFO:0030031",  # SCOPE-chip
    "EFO:0030059",  # 10x multiome
    "EFO:0030060",  # mCT-seq
    "EFO:0030061",  # mcSCRB-seq
    "EFO:0030074",  # SORT-seq
    "EFO:0700003",  # BD Rhapsody Whole Transcriptome Analysis
    "EFO:0700004",  # BD Rhapsody Targeted mRNA
    "EFO:0700010",  # TruDrop
    "EFO:0700011",  # GEXSCOPE technology
    "EFO:0700016",  # Smart-seq v4
    "EFO:0900000",  # particle-templated instant partition sequencing
    "EFO:0900001",  # Asteria scRNA-seq kit
    "EFO:0900002",  # HIVE CLX Single-Cell RNAseq Solution
]

# list of EFO terms that correspond to SPATIAL modality/measurement. These terms
# define the inclusive filter applied to obs.assay_ontology_term_id. All other
ALLOWED_SPATIAL_ASSAYS = [
    "EFO:0022857",  # Visium Spatial Gene Expression V1
    "EFO:0030062",  # Slide-seqV2
]

# Full-gene assays have special handling in the "normalized" X layers
FULL_GENE_ASSAY = [
    "EFO:0003755",  # FL-cDNA
    "EFO:0008747",  # FRISCR
    "EFO:0008763",  # Hi-SCL
    "EFO:0008797",  # MATQ-seq
    "EFO:0008877",  # Quartz-seq
    "EFO:0008930",  # Smart-seq
    "EFO:0008931",  # Smart-seq2
    "EFO:0008956",  # SUPeR-seq
    "EFO:0009999",  # G&T-Seq
    "EFO:0010004",  # SCRB-seq
    "EFO:0010006",  # scM&T-seq
    "EFO:0010022",  # Smart-3Seq
    "EFO:0010058",  # Fluidigm C1-based SMARTer library preparation
    "EFO:0010184",  # Smart-like
    "EFO:0022396",  # TruSeq
    "EFO:0022488",  # Smart-seq3
    "EFO:0022839",  # STORM-seq
    "EFO:0030031",  # SCOPE-chip
    "EFO:0030060",  # mCT-seq
    "EFO:0030061",  # mcSCRB-seq
    "EFO:0700016",  # Smart-seq v4
]


# The default configuration for TileDB contexts used in the builder.
# Ref: https://docs.tiledb.com/main/how-to/configuration#configuration-parameters
DEFAULT_TILEDB_CONFIG = {
    "py.init_buffer_bytes": 1 * 1024**3,
    "py.deduplicate": "true",
    "soma.init_buffer_bytes": 1 * 1024**3,
    "sm.mem.reader.sparse_global_order.ratio_array_data": 0.3,
    #
    # Concurrency levels are capped for high-CPU boxes. Left unchecked,
    # the default configs will hit kernel limits on high-CPU boxes. This
    # cap can be raised when TiledB-SOMA is more thread frugal. See for
    # example: https://github.com/single-cell-data/TileDB-SOMA/issues/2149
    "sm.compute_concurrency_level": min(cpu_count(), 48),
    "sm.io_concurrency_level": min(cpu_count(), 48),
}


"""
Singletons used throughout the package
"""


@functools.cache
def SOMA_TileDB_Context() -> soma.options.SOMATileDBContext:
    return soma.options.SOMATileDBContext(tiledb_config=DEFAULT_TILEDB_CONFIG, timestamp=None)
