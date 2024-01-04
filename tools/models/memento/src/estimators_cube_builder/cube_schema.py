from tiledb import ArraySchema, Attr, ByteShuffleFilter, DictionaryFilter, Dim, Domain, FilterList, ZstdFilter

CUBE_TILEDB_DIMS_OBS = [
    "cell_type",
    "dataset_id",
]

CUBE_TILEDB_ATTRS_OBS = ["assay", "suspension_type", "donor_id", "disease", "sex"]

CUBE_LOGICAL_DIMS_OBS = CUBE_TILEDB_DIMS_OBS + CUBE_TILEDB_ATTRS_OBS

CUBE_DIMS_VAR = ["feature_id"]

CUBE_TILEDB_DIMS = CUBE_TILEDB_DIMS_OBS + CUBE_DIMS_VAR

ESTIMATOR_NAMES = ["nnz", "n_obs", "min", "max", "sum", "mean", "sem", "var", "sev", "selv"]


CUBE_SCHEMA = ArraySchema(
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
