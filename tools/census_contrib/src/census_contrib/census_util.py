from __future__ import annotations

import functools
from typing import Optional, Tuple, cast

import cellxgene_census
import numpy as np
import numpy.typing as npt
import tiledbsoma as soma

from .config import Config
from .util import get_logger

logger = get_logger()


def open_census(census_version: Optional[str], census_uri: Optional[str]) -> soma.Collection:
    """Open and return the Census top-level handle."""

    if census_uri:
        return cellxgene_census.open_soma(uri=census_uri)

    return cellxgene_census.open_soma(census_version=census_version)


@functools.cache
def get_axis_soma_joinids(config: Config) -> Tuple[npt.NDArray[np.int64], Tuple[int, ...]]:
    """
    Return experiment axis (obs, var) soma_joind values and axis shape appropriate for the
    Census version specified in the metadata and the axis specified in embedding metadata.
    """
    axis: str = config.metadata.data_type
    assert axis in [
        "obs_embedding",
        "var_embedding",
    ], "Unknown embedding type - must be one of obs_embedding or var_embedding"

    with open_census(census_uri=config.args.census_uri, census_version=config.metadata.census_version) as census:
        exp = census["census_data"][config.metadata.experiment_name]
        if axis == "obs_embedding":
            tbl = exp.obs.read(column_names=["soma_joinid"]).concat()
        else:
            tbl = exp.ms["RNA"].var.read(column_names=["soma_joinid"]).concat()

        joinids = cast(npt.NDArray[np.int64], tbl.column("soma_joinid").to_numpy())
        return joinids, (joinids.max() + 1,)


def get_census_obs_uri_region(config: Config) -> Tuple[str, str]:
    with open_census(census_uri=config.args.census_uri, census_version=config.metadata.census_version) as census:
        exp = census["census_data"][config.metadata.experiment_name]
        uri = exp.obs.uri
        assert isinstance(uri, str)

    if not config.args.census_uri:
        desc = cellxgene_census.get_census_version_description(census_version=config.metadata.census_version)
        region = desc.get("soma", {}).get("s3_region", "us-west-2")
    else:
        region = None

    return uri, region
