from __future__ import annotations

import pandas as pd
import tiledbsoma as soma


def mean_variance(
    query: soma.ExperimentAxisQuery,
    layer: str = "raw",
    calculate_mean: bool = False,
    calculate_variance: bool = False,
) -> pd.DataFrame:
    """
    Calculate  mean and/or variance along the `obs` axis from query results. Calculations
    are done in an accumulative chunked fashion.

    Args:
        query:
            A SOMA query, specifying the obs/var selection over which mean and variance are calculated.

        layer:
            X layer used, e.g., `raw`

        calculate_mean:
            If `True it calculates mean, otherwise skips calculation

        calculate_variance:
            If `True it calculates variance, otherwise skips calculation

    Returns:
        Pandas DataFrame indexed by the obs `soma_joinid` and with columns
       `mean` (if `calculate_mean = True`), and `variance` (if `calculate_variance = True`)

    Lifecycle:
        experimental
    """
    return None
