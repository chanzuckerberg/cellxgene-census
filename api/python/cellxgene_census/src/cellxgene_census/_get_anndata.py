# Copyright (c) 2022-2023 Chan Zuckerberg Initiative
#
# Licensed under the MIT License.

"""Get slice as AnnData

Methods to retrieve slices of the census as AnnData objects.
"""
from typing import Optional

import anndata
import tiledbsoma as soma
from somacore.options import SparseDFCoord

from ._experiment import _get_experiment


def get_anndata(
    census: soma.Collection,
    organism: str,
    measurement_name: str = "RNA",
    X_name: str = "raw",
    obs_value_filter: Optional[str] = None,
    obs_coords: Optional[SparseDFCoord] = None,
    var_value_filter: Optional[str] = None,
    var_coords: Optional[SparseDFCoord] = None,
    column_names: Optional[soma.AxisColumnNames] = None,
) -> anndata.AnnData:
    """
    Convience wrapper around ``soma.Experiment`` query, to build and execute a query,
    and return it as an :class:`anndata.AnnData` object.

    Args:
        census:
            The census object, usually returned by :func:`cellxgene_census.open_soma()`.
        organism:
            The organism to query, usually one of `Homo sapiens` or `Mus musculus`.
        measurement_name:
            The measurement object to query. Defaults to `RNA`.
        X_name:
            The ``X`` layer to query. Defaults to `raw`.
        obs_value_filter:
            Value filter for the ``obs`` metadata. Value is a filter query written in the
            SOMA ``value_filter`` syntax.
        obs_coords:
            Coordinates for the ``obs`` axis, which is indexed by the ``soma_joinid`` value.
            May be an ``int``, a list of ``int``, or a slice. The default, ``None``, selects all.
        var_value_filter:
            Value filter for the ``var`` metadata. Value is a filter query written in the
            SOMA ``value_filter`` syntax.
        var_coords:
            Coordinates for the ``var`` axis, which is indexed by the ``soma_joinid`` value.
            May be an ``int``, a list of ``int``, or a slice. The default, ``None``, selects all.
        column_names:
            Columns to fetch for ``obs`` and ``var`` dataframes.

    Returns:
        An :class:`anndata.AnnData` object containing the census slice.

    Lifecycle:
        maturing

    Examples:
        >>> get_anndata(census, "Mus musculus", obs_value_filter="tissue_general in ['brain', 'lung']")

        >>> get_anndata(census, "Homo sapiens", column_names={"obs": ["tissue"]})

        >>> get_anndata(census, "Homo sapiens", obs_coords=slice(0, 1000))
    """
    exp = _get_experiment(census, organism)
    obs_coords = (slice(None),) if obs_coords is None else (obs_coords,)
    var_coords = (slice(None),) if var_coords is None else (var_coords,)
    with exp.axis_query(
        measurement_name,
        obs_query=soma.AxisQuery(value_filter=obs_value_filter, coords=obs_coords),
        var_query=soma.AxisQuery(value_filter=var_value_filter, coords=var_coords),
    ) as query:
        return query.to_anndata(X_name=X_name, column_names=column_names)
