from typing import Optional

import anndata
import tiledbsoma as soma
from typing_extensions import TypedDict

from .experiment import get_experiment
from .experiment_query import AxisColumnNames, AxisQuery, experiment_query

AxisValueFilters = TypedDict(
    "AxisValueFilters",
    {
        "obs": Optional[str],
        "var": Optional[str],
    },
)


def get_anndata(
    census: soma.Collection,
    organism: str,
    measurement_name: str = "RNA",
    X_name: str = "raw",
    value_filter: Optional[AxisValueFilters] = None,
    column_names: Optional[AxisColumnNames] = None,
) -> anndata.AnnData:
    """
    Convience wrapper around soma.Experiment query, to build and execute a query,
    and return it as an AnnData object.

    Parameters
    ----------
    census : soma.Collection
        The census object, usually returned by `cell_census.open_soma()`
    organism : str
        The organism to query, usually one of "Homo sapiens" or "Mus musculus"
    measurement_name : str, default 'RNA'
        The measurement object to query
    X_name : str, default "raw"
        The X layer to query
    value_filter : dict[Literal['obs', 'var'], str]
        Value filter definition for ``obs`` and ``var`` metadata. Value is a filter query
        written in the SOMA ``value_filter`` syntax.
    column_names: dict[Literal['obs', 'var'], List[str]]
        Colums to fetch for obs and var dataframes.

    Returns
    -------
    anndata.AnnData - containing the census slice

    Examples
    --------
    >>> get_anndata(census, "Mus musculus", value_filter={"obs": "tissue_general in ['brain', 'lung']"})

    >>> get_anndata(census, "Homo sapiens", column_names={"obs": ["tissue"]})

    """
    exp = get_experiment(census, organism)
    _obs_value_filter = None if value_filter is None else value_filter.get("obs", None)
    _var_value_filter = None if value_filter is None else value_filter.get("var", None)
    with experiment_query(
        exp,
        measurement_name=measurement_name,
        obs_query=AxisQuery(value_filter=_obs_value_filter) if _obs_value_filter is not None else None,
        var_query=AxisQuery(value_filter=_var_value_filter) if _var_value_filter is not None else None,
    ) as query:
        return query.read_as_anndata(X_name=X_name, column_names=column_names)
