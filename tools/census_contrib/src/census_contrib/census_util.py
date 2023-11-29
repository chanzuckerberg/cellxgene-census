from __future__ import annotations

import functools
from typing import Tuple, cast

import cellxgene_census
import numpy as np
import numpy.typing as npt

from .metadata import EmbeddingMetadata
from .util import get_logger

logger = get_logger()


@functools.cache
def get_obs_soma_joinids(metadata: EmbeddingMetadata) -> Tuple[npt.NDArray[np.int64], Tuple[int, ...]]:
    """
    Return experiment obs soma_joind values and obs shape appropriate for the
    Census version specified in the metadata.
    """
    logger.info(f"Loading obs joinids from census_version={metadata.census_version}")
    with cellxgene_census.open_soma(census_version=metadata.census_version) as census:
        exp = census["census_data"][metadata.experiment_name]
        tbl = exp.obs.read(column_names=["soma_joinid"]).concat()

        joinids = cast(npt.NDArray[np.int64], tbl.column("soma_joinid").to_numpy())
        return joinids, (joinids.max() + 1,)


def get_census_obs_uri_region(metadata: EmbeddingMetadata) -> Tuple[str, str]:
    with cellxgene_census.open_soma(census_version=metadata.census_version) as census:
        exp = census["census_data"][metadata.experiment_name]
        uri = exp.obs.uri
        assert isinstance(uri, str)

    desc = cellxgene_census.get_census_version_description(census_version=metadata.census_version)
    region = desc.get("soma", {}).get("s3_region", "us-west-2")
    return uri, region
