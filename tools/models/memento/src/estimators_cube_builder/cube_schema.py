from tiledb import ArraySchema, Attr, ByteShuffleFilter, DictionaryFilter, Dim, Domain, FilterList, ZstdFilter

CUBE_LOGICAL_DIMS_OBS = [
    "cell_type_ontology_term_id",
    "dataset_id",
    "tissue_general_ontology_term_id",
    "assay_ontology_term_id",
    "donor_id",
    "disease_ontology_term_id",
    "sex_ontology_term_id",
    "development_stage_ontology_term_id",
    "self_reported_ethnicity_ontology_term_id",
    "suspension_type",
]

CUBE_TILEDB_DIMS_OBS = [CUBE_LOGICAL_DIMS_OBS[0]]

CUBE_TILEDB_ATTRS_OBS = CUBE_LOGICAL_DIMS_OBS[1:]

CUBE_DIMS_VAR = ["feature_id"]

CUBE_TILEDB_DIMS = CUBE_TILEDB_DIMS_OBS + CUBE_DIMS_VAR

# ESTIMATOR_NAMES = ["nnz", "n_obs", "min", "max", "sum", "mean", "sem", "var", "sev", "selv"]
ESTIMATOR_NAMES = ["n_obs", "mean", "sem", "var", "selv"]


def build_cube_schema() -> ArraySchema:
    return ArraySchema(
        domain=Domain(
            *[
                Dim(name=dim_name, dtype="ascii", filters=FilterList([DictionaryFilter(), ZstdFilter(level=19)]))
                for dim_name in CUBE_TILEDB_DIMS
            ]
        ),
        attrs=[
            Attr(
                name=attr_name,
                dtype="ascii",
                nullable=False,
                filters=FilterList([DictionaryFilter(), ZstdFilter(level=19)]),
            )
            for attr_name in CUBE_TILEDB_ATTRS_OBS
        ]
        + [
            Attr(
                name=estimator_name,
                dtype="float64",
                var=False,
                nullable=False,
                filters=FilterList([ByteShuffleFilter(), ZstdFilter(level=5)]),
            )
            for estimator_name in ESTIMATOR_NAMES
        ],
        cell_order="row-major",
        tile_order="row-major",
        capacity=10000,
        sparse=True,
        allows_duplicates=True,
    )
