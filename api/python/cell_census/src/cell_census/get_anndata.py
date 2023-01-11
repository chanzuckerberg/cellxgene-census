from typing import Optional

import anndata
import tiledbsoma as soma

from .experiment import get_experiment


def get_anndata(
    census: soma.Collection,
    organism: str,
    measurement_name: str = "RNA",
    X_name: str = "raw",
    obs_value_filter: Optional[str] = None,
    var_value_filter: Optional[str] = None,
    column_names: Optional[soma.experiment_query.AxisColumnNames] = None,
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
    obs_value_filter: str, default None
        Value filter for the ``obs`` metadata. Value is a filter query written in the
        SOMA ``value_filter`` syntax.
    var_value_filter: str, default None
        Value filter for the ``var`` metadata. Value is a filter query written in the
        SOMA ``value_filter`` syntax.
    column_names: dict[Literal['obs', 'var'], List[str]]
        Colums to fetch for obs and var dataframes.

    Returns
    -------
    anndata.AnnData - containing the census slice

    Examples
    --------
    >>> get_anndata(census, "Mus musculus", obs_value_filter="tissue_general in ['brain', 'lung']")

    >>> get_anndata(census, "Homo sapiens", column_names={"obs": ["tissue"]})

    """
    exp = get_experiment(census, organism)
    with exp.axis_query(
        measurement_name,
        obs_query=soma.experiment_query.AxisQuery(value_filter=obs_value_filter)
        if obs_value_filter is not None
        else None,
        var_query=soma.experiment_query.AxisQuery(value_filter=var_value_filter)
        if var_value_filter is not None
        else None,
    ) as query:
        return query.to_anndata(X_name=X_name, column_names=column_names)
