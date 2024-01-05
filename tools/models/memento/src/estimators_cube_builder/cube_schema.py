from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from tiledb import (
    ArraySchema,
    Attr,
    ByteShuffleFilter,
    DictionaryFilter,
    Dim,
    Domain,
    DoubleDeltaFilter,
    Enumeration,
    FilterList,
    ZstdFilter,
)

OBS_TILEDB_DIMS = ["obs_group_joinid"]

OBS_LOGICAL_DIMS = [
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

CUBE_LOGICAL_DIMS = ["feature_id"] + OBS_LOGICAL_DIMS

ESTIMATORS_TILEDB_DIMS = ["obs_group_joinid", "feature_id"]

# ESTIMATOR_NAMES = ["nnz", "n_obs", "min", "max", "sum", "mean", "sem", "var", "sev", "selv"]
ESTIMATOR_NAMES = ["n_obs", "mean", "sem"]


def build_obs_categorical_values(obs_groups: pd.DataFrame) -> Dict[str, Enumeration]:
    return {dim_name: obs_groups[dim_name].unique().astype(str) for dim_name in OBS_LOGICAL_DIMS}


def build_obs_groups_schema(n_obs_groups: int, obs_categorical_values: Dict[str, Enumeration]) -> ArraySchema:
    domain = Domain(
        Dim(
            name="obs_group_joinid",
            dtype=np.uint32,
            domain=(0, n_obs_groups),
            filters=FilterList([ZstdFilter(level=19)]),
        )
    )
    assert set(OBS_TILEDB_DIMS) == set([dim.name for dim in domain])
    return ArraySchema(
        enums=[
            Enumeration(name=dim_name, ordered=False, values=categories)
            for (dim_name, categories) in obs_categorical_values.items()
        ],
        domain=domain,
        # TODO: Not all attrs need to be int32
        attrs=[
            Attr(
                name=attr_name,
                dtype=np.int32,
                enum_label=attr_name,
                nullable=False,
                filters=FilterList([ZstdFilter(level=19)]),
            )
            for attr_name in OBS_LOGICAL_DIMS
        ],
        offsets_filters=FilterList([DoubleDeltaFilter(), ZstdFilter(level=19)]),
        cell_order="row-major",
        tile_order="row-major",
        capacity=10000,
        sparse=True,  # TODO: Dense would work
        allows_duplicates=True,
    )


def build_estimators_schema(n_groups: int) -> ArraySchema:
    domain = Domain(
        Dim(name="obs_group_joinid", dtype=np.uint32, domain=(0, n_groups), filters=FilterList([ZstdFilter(level=19)])),
        Dim(name="feature_id", dtype="ascii", filters=FilterList([DictionaryFilter(), ZstdFilter(level=19)])),
    )
    assert ESTIMATORS_TILEDB_DIMS == [dim.name for dim in domain]
    return ArraySchema(
        domain=domain,
        attrs=[
            Attr(
                name=estimator_name,
                # TODO: use float32?
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


OBS_GROUPS_ARRAY = "obs_groups"
ESTIMATORS_ARRAY = "estimators"
