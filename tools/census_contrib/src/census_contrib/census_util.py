from __future__ import annotations

import functools
from typing import Tuple, cast

import cellxgene_census
import numpy as np
import numpy.typing as npt

from .metadata import EmbeddingMetadata


@functools.cache
def get_obs_soma_joinids(
    metadata: EmbeddingMetadata,
) -> Tuple[npt.NDArray[np.int64], Tuple[int, ...]]:
    """
    Return experiment obs soma_joind values and obs shape appropriate for the
    Census version specified in the metadata.
    """
    with cellxgene_census.open_soma(census_version=metadata.census_version) as census:
        exp = census["census_data"][metadata.experiment_name]
        tbl = exp.obs.read(column_names=["soma_joinid"]).concat()

        joinids = cast(npt.NDArray[np.int64], tbl.column("soma_joinid").to_numpy())
        return joinids, (joinids.max() + 1,)
