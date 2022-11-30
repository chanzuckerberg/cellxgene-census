"""
Types global to this module
"""
from typing import Optional, Sequence, TypedDict

import pandas as pd
import pyarrow as pa

# Sadly, you can't define a generic TypedDict....


class ExperimentQueryReadArrowResult(TypedDict, total=False):
    obs: pa.Table
    var: pa.Table
    X: pa.Table
    X_layers: dict[str, pa.Table]


class ExperimentQueryReadPandasResult(TypedDict, total=False):
    obs: pd.DataFrame
    var: pd.DataFrame
    X: pd.DataFrame
    X_layers: dict[str, pd.DataFrame]


AxisColumnNames = TypedDict(
    "AxisColumnNames",
    {
        "obs": Optional[Sequence[str]],  # None is all
        "var": Optional[Sequence[str]],
    },
)
