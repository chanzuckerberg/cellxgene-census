import numbers
import re
from typing import List, Optional, Union

import anndata
import tiledbsoma as soma
from typing_extensions import TypedDict

from .experiment_query import AxisColumnNames, AxisQuery, experiment_query

ObsQuery = TypedDict(
    "ObsQuery",
    {
        "assay": Optional[Union[str, List[str]]],
        "assay_ontology_term_id": Optional[Union[str, List[str]]],
        "cell_type": Optional[Union[str, List[str]]],
        "cell_type_ontology_term_id": Optional[Union[str, List[str]]],
        "development_stage": Optional[Union[str, List[str]]],
        "development_stage_ontology_term_id": Optional[Union[str, List[str]]],
        "disease": Optional[Union[str, List[str]]],
        "disease_ontology_term_id": Optional[Union[str, List[str]]],
        "donor_id": Optional[Union[str, List[str]]],
        "is_primary_data": Optional[bool],
        "self_reported_ethnicity": Optional[Union[str, List[str]]],
        "self_reported_ethnicity_ontology_term_id": Optional[Union[str, List[str]]],
        "sex": Optional[Union[str, List[str]]],
        "sex_ontology_term_id": Optional[Union[str, List[str]]],
        "suspension_type": Optional[Union[str, List[str]]],
        "tissue": Optional[Union[str, List[str]]],
        "tissue_ontology_term_id": Optional[Union[str, List[str]]],
        "tissue_general": Optional[Union[str, List[str]]],
        "tissue_general_ontology_term_id": Optional[Union[str, List[str]]],
    },
)

VarQuery = TypedDict(
    "VarQuery",
    {
        "feature_id": Optional[Union[str, List[str]]],
        "feature_name": Optional[Union[str, List[str]]],
    },
)


def _build_query(query_defn: Optional[Union[ObsQuery, VarQuery]] = None) -> Optional[AxisQuery]:
    """
    Build a AxisQuery value filter from the user-defined query parameters.
    """
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
    obs_query: Optional[ObsQuery] = None,
    var_query: Optional[VarQuery] = None,
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
    obs_query : dict[str, Union[str, List[str]]]
        Obs (cell) query definition. Dict where keys are column names, and value is a
        string or list of strings to match. All query terms must match (AND query).
    var_query : dict[str, Union[str, List[str]]]
        Var (gene) query definition. Dict where keys are column names, and value is a
        string or list of strings to match. All query terms must match (AND query).
    column_names: dict[Literal['obs', 'var'], List[str]]
        Colums to fetch for obs and var dataframes.

    Returns
    -------
    anndata.AnnData - containing the census slice

    Examples
    --------
    >>> get_anndata(census, "Mus musculus", obs_query={"tissue": "brain"})

    >>> get_anndata(census, "Homo sapiens", column_names={"obs": ["tissue"]})

    """

    # lower/snake case the organism name to find the experiment name
    exp_name = re.sub(r"[ ]+", "_", organism).lower()

    if exp_name not in census["census_data"]:
        raise ValueError(f"Unknown organism {organism} - does not exist")
    exp = census["census_data"][exp_name]
    if exp.soma_type != "SOMAExperiment":
        raise ValueError(f"Unknown organism {organism} - not a SOMA Experiment")

    _obs_query = _build_query(obs_query)
    _var_query = _build_query(var_query)
    with experiment_query(exp, measurement_name=measurement_name, obs_query=_obs_query, var_query=_var_query) as query:
        return query.read_as_anndata(X_name=X_name, column_names=column_names)
