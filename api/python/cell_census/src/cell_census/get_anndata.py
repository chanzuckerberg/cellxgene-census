import numbers
from typing import List, Optional, Union

import anndata
import tiledbsoma as soma
from typing_extensions import NotRequired, TypedDict

from .experiment import get_experiment
from .experiment_query import AxisColumnNames, AxisCoordinate, AxisQuery, experiment_query

ObsQuery = TypedDict(
    "ObsQuery",
    {
        "assay": NotRequired[Union[str, List[str]]],
        "assay_ontology_term_id": NotRequired[Union[str, List[str]]],
        "cell_type": NotRequired[Union[str, List[str]]],
        "cell_type_ontology_term_id": NotRequired[Union[str, List[str]]],
        "development_stage": NotRequired[Union[str, List[str]]],
        "development_stage_ontology_term_id": NotRequired[Union[str, List[str]]],
        "disease": NotRequired[Union[str, List[str]]],
        "disease_ontology_term_id": NotRequired[Union[str, List[str]]],
        "donor_id": NotRequired[Union[str, List[str]]],
        "is_primary_data": NotRequired[bool],
        "self_reported_ethnicity": NotRequired[Union[str, List[str]]],
        "self_reported_ethnicity_ontology_term_id": NotRequired[Union[str, List[str]]],
        "sex": NotRequired[Union[str, List[str]]],
        "sex_ontology_term_id": NotRequired[Union[str, List[str]]],
        "suspension_type": NotRequired[Union[str, List[str]]],
        "tissue": NotRequired[Union[str, List[str]]],
        "tissue_ontology_term_id": NotRequired[Union[str, List[str]]],
        "tissue_general": NotRequired[Union[str, List[str]]],
        "tissue_general_ontology_term_id": NotRequired[Union[str, List[str]]],
    },
)

VarQuery = TypedDict(
    "VarQuery",
    {
        "feature_id": NotRequired[Union[str, List[str]]],
        "feature_name": NotRequired[Union[str, List[str]]],
    },
)


def _build_query(
    joinids: Optional[AxisCoordinate], query_defn: Optional[Union[ObsQuery, VarQuery]] = None
) -> Optional[AxisQuery]:
    """
    Build a AxisQuery value filter from the user-defined query parameters.
    """
    if joinids is not None and query_defn is not None:
        raise ValueError("Only one of joinids or value filter may be specified in a query.")

    if joinids is not None:
        return AxisQuery(coords=(joinids,))

    if query_defn is None:
        return None

    query_conditions = []
    for name, val in query_defn.items():
        if isinstance(val, str):
            query_conditions.append(f"{name} == '{val}'")
        elif isinstance(val, numbers.Number):
            query_conditions.append(f"{name} == {val}")
        elif isinstance(val, list):
            query_conditions.append(f"{name} in {val}")
        else:
            raise TypeError("Query must be string or list of strings")

    if len(query_conditions) == 0:
        return None

    return AxisQuery(value_filter=" and ".join(query_conditions))


def get_anndata(
    census: soma.Collection,
    organism: str,
    measurement_name: str = "RNA",
    X_name: str = "raw",
    obs_joinids: Optional[AxisCoordinate] = None,
    obs_value_filter: Optional[ObsQuery] = None,
    var_joinids: Optional[AxisCoordinate] = None,
    var_value_filter: Optional[VarQuery] = None,
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
    obs_joinids : Union[slice, int, ArrayLike]
        Obs (cell) slice, defined as ``soma_joinid``, slice of joinids, or a NumPy ArrayLike of joinids.
        Note: only one of ``obs_joinids`` and ``obs_value_filter`` may be specified.
    obs_value_filter : dict[str, Union[str, List[str]]]
        Obs (cell) value filter definition. Dict where keys are column names, and value is a
        string or list of strings to match. All query terms must match (AND query).
        Note: only one of ``obs_joinids`` and ``obs_value_filter`` may be specified.
    var_joinids : Union[slice, int, ArrayLike]
        Var (gene) slice, defined as ``soma_joinid``, slice of joinids, or a NumPy ArrayLike of joinids.
        Note: only one of ``var_joinids`` and ``var_value_filter`` may be specified.
    var_value_filter : dict[str, Union[str, List[str]]]
        Var (gene) value filter definition. Dict where keys are column names, and value is a
        string or list of strings to match. All query terms must match (AND query).
        Note: only one of ``var_joinids`` and ``var_value_filter`` may be specified.
    column_names: dict[Literal['obs', 'var'], List[str]]
        Colums to fetch for obs and var dataframes.

    Returns
    -------
    anndata.AnnData - containing the census slice

    Examples
    --------
    >>> get_anndata(census, "Mus musculus", obs_value_filter={"tissue": "brain"})

    >>> get_anndata(census, "Homo sapiens", column_names={"obs": ["tissue"]})

    >>> get_anndata(cesus, "Homo sapiens", obs_joinds=[0,88,222])

    """
    exp = get_experiment(census, organism)
    _obs_query = _build_query(obs_joinids, obs_value_filter)
    _var_query = _build_query(var_joinids, var_value_filter)
    with experiment_query(
        exp,
        measurement_name=measurement_name,
        obs_query=_obs_query,
        var_query=_var_query,
    ) as query:
        return query.read_as_anndata(X_name=X_name, column_names=column_names)
