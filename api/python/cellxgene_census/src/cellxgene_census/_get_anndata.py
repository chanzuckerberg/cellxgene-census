# Copyright (c) 2022, Chan Zuckerberg Initiative
#
# Licensed under the MIT License.

"""Get slice as AnnData.

Methods to retrieve slices of the census as AnnData objects.
"""

from collections.abc import Sequence
from typing import Literal
from warnings import warn

import anndata
import pandas as pd
import tiledbsoma as soma
from somacore.options import SparseDFCoord

from ._experiment import _get_experiment, _get_experiment_name
from ._release_directory import get_census_version_directory
from ._util import _extract_census_version, _uri_join

CENSUS_EMBEDDINGS_LOCATION_BASE_URI = "s3://cellxgene-contrib-public/contrib/cell-census/soma/"


def get_anndata(
    census: soma.Collection,
    organism: str,
    measurement_name: str = "RNA",
    X_name: str = "raw",
    X_layers: Sequence[str] | None = (),
    obsm_layers: Sequence[str] | None = (),
    obsp_layers: Sequence[str] | None = (),
    varm_layers: Sequence[str] | None = (),
    varp_layers: Sequence[str] | None = (),
    obs_value_filter: str | None = None,
    obs_coords: SparseDFCoord | None = None,
    var_value_filter: str | None = None,
    var_coords: SparseDFCoord | None = None,
    column_names: soma.AxisColumnNames | None = None,
    obs_embeddings: Sequence[str] | None = (),
    var_embeddings: Sequence[str] | None = (),
    obs_column_names: Sequence[str] | None = None,
    var_column_names: Sequence[str] | None = None,
    modality: str = "census_data",
) -> anndata.AnnData:
    """Convenience wrapper around :class:`tiledbsoma.Experiment` query, to build and execute a query,
    and return it as an :class:`anndata.AnnData` object.

    Args:
        census:
            The census object, usually returned by :func:`open_soma`.
        organism:
            The organism to query, usually one of ``"Homo sapiens`` or ``"Mus musculus"``.
        measurement_name:
            The measurement object to query. Defaults to ``"RNA"``.
        X_name:
            The ``X`` layer to query. Defaults to ``"raw"``.
        X_layers:
            Additional layers to add to :attr:`anndata.AnnData.layers`.
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
        obsm_layers:
            Additional obsm layers to read and return in the ``obsm`` slot.
        obsp_layers:
            Additional obsp layers to read and return in the ``obsp`` slot.
        varm_layers:
            Additional varm layers to read and return in the ``varm`` slot.
        varp_layers:
            Additional varp layers to read and return in the ``varp`` slot.
        obs_embeddings:
            Additional embeddings to be returned as part of the ``obsm`` slot.
            Use :func:`get_all_available_embeddings` to retrieve available embeddings
            for this Census version and organism.
        var_embeddings:
            Additional embeddings to be returned as part of the ``varm`` slot.
            Use :func:`get_all_available_embeddings` to retrieve available embeddings
            for this Census version and organism.
        obs_column_names:
            Columns to fetch for ``obs`` dataframe.
        var_column_names:
            Columns to fetch for ``var`` dataframe.
        modality:
            Which modality to query, can be one of ``"census_data"`` or ``"census_spatial_sequencing"``.
            Defaults to ``"census_data"``.

    Returns:
        An :class:`anndata.AnnData` object containing the census slice.

    Lifecycle:
        experimental

    Examples:
        >>> get_anndata(census, "Mus musculus", obs_value_filter="tissue_general in ['brain', 'lung']")

        >>> get_anndata(census, "Homo sapiens", obs_column_names=["tissue"])

        >>> get_anndata(census, "Homo sapiens", obs_coords=slice(0, 1000))
    """
    exp = _get_experiment(census, organism, modality)
    obs_coords = (slice(None),) if obs_coords is None else (obs_coords,)
    var_coords = (slice(None),) if var_coords is None else (var_coords,)

    if obsm_layers and obs_embeddings and set(obsm_layers) & set(obs_embeddings):
        raise ValueError("Cannot request both `obsm_layers` and `obs_embeddings` for the same embedding name")

    if varm_layers and var_embeddings and set(varm_layers) & set(var_embeddings):
        raise ValueError("Cannot request both `varm_layers` and `var_embeddings` for the same embedding name")

    # Backwards compat for old column_names argument
    if column_names is not None:
        if obs_column_names is not None or var_column_names is not None:
            raise ValueError(
                "Both the deprecated 'column_names' argument and its replacements were used. Please use 'obs_column_names' and 'var_column_names' only."
            )
        else:
            warn(
                "The argument `column_names` is deprecated and will be removed in a future release. Please use `obs_column_names` and `var_column_names` instead.",
                FutureWarning,
                stacklevel=2,
            )
        if "obs" in column_names:
            obs_column_names = column_names["obs"]
        if "var" in column_names:
            var_column_names = column_names["var"]

    with exp.axis_query(
        measurement_name,
        obs_query=soma.AxisQuery(value_filter=obs_value_filter, coords=obs_coords),
        var_query=soma.AxisQuery(value_filter=var_value_filter, coords=var_coords),
    ) as query:
        adata = query.to_anndata(
            X_name=X_name,
            column_names={"obs": obs_column_names, "var": var_column_names},
            X_layers=X_layers,
            obsm_layers=obsm_layers,
            varm_layers=varm_layers,
            obsp_layers=obsp_layers,
            varp_layers=varp_layers,
        )

        # If obs_embeddings or var_embeddings are defined, inject them in the appropriate slot
        if obs_embeddings or var_embeddings:
            if modality == "census_spatial_sequencing":
                raise ValueError("Embeddings are not supported for the spatial sequencing collection at this time.")

            from .experimental._embedding import _get_embedding, get_embedding_metadata_by_name

            census_version = _extract_census_version(census)
            experiment_name = _get_experiment_name(organism)
            census_directory = get_census_version_directory()

            if obs_embeddings:
                obs_soma_joinids = query.obs_joinids()
                for emb in obs_embeddings:
                    emb_metadata = get_embedding_metadata_by_name(emb, experiment_name, census_version, "obs_embedding")
                    uri = _uri_join(CENSUS_EMBEDDINGS_LOCATION_BASE_URI, f"{census_version}/{emb_metadata['id']}")
                    embedding = _get_embedding(census, census_directory, census_version, uri, obs_soma_joinids)
                    adata.obsm[emb] = embedding

            if var_embeddings:
                var_soma_joinids = query.var_joinids()
                for emb in var_embeddings:
                    emb_metadata = get_embedding_metadata_by_name(emb, experiment_name, census_version, "var_embedding")
                    uri = _uri_join(CENSUS_EMBEDDINGS_LOCATION_BASE_URI, f"{census_version}/{emb_metadata['id']}")
                    embedding = _get_embedding(census, census_directory, census_version, uri, var_soma_joinids)
                    adata.varm[emb] = embedding

        return adata


def _get_axis_metadata(
    census: soma.Collection,
    axis: Literal["obs", "var"],
    organism: str,
    modality: str = "census_data",
    *,
    value_filter: str | None = None,
    coords: SparseDFCoord | None = slice(None),
    column_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    exp = _get_experiment(census, organism, modality=modality)
    coords = (slice(None),) if coords is None else (coords,)
    if axis == "obs":
        df = exp.obs
    elif axis == "var":
        df = exp.ms["RNA"].var
    else:
        raise ValueError(f"axis should be either 'obs' or 'var', but '{axis}' was passed")
    result: pd.DataFrame = (
        df.read(coords=coords, column_names=column_names, value_filter=value_filter).concat().to_pandas()
    )
    return result


def get_obs(
    census: soma.Collection,
    organism: str,
    *,
    value_filter: str | None = None,
    coords: SparseDFCoord | None = slice(None),
    column_names: Sequence[str] | None = None,
    modality: str = "census_data",
) -> pd.DataFrame:
    """Get the observation metadata for a query on the census.

    Args:
        census:
            The census object, usually returned by :func:`open_soma`.
        organism:
            The organism to query, usually one of ``"Homo sapiens`` or ``"Mus musculus"``
        value_filter:
            Value filter for the ``obs`` metadata. Value is a filter query written in the
            SOMA ``value_filter`` syntax.
        coords:
            Coordinates for the ``obs`` axis, which is indexed by the ``soma_joinid`` value.
            May be an ``int``, a list of ``int``, or a slice. The default, ``None``, selects all.
        column_names:
            Columns to fetch.
        modality
            Which modality to query, can be one of ``"census_data"`` or ``"census_spatial_sequencing"``.
            Defaults to ``"census_data"``.

    Returns:
        A :class:`pandas.DataFrame` object containing metadata for the queried slice.
    """
    return _get_axis_metadata(
        census,
        "obs",
        organism,
        value_filter=value_filter,
        coords=coords,
        column_names=column_names,
        modality=modality,
    )


def get_var(
    census: soma.Collection,
    organism: str,
    *,
    value_filter: str | None = None,
    coords: SparseDFCoord | None = slice(None),
    column_names: Sequence[str] | None = None,
    modality: str = "census_data",
) -> pd.DataFrame:
    """Get the variable metadata for a query on the census.

    Args:
        census:
            The census object, usually returned by :func:`open_soma`.
        organism:
            The organism to query, usually one of ``"Homo sapiens`` or ``"Mus musculus"``
        value_filter:
            Value filter for the ``var`` metadata. Value is a filter query written in the
            SOMA ``value_filter`` syntax.
        coords:
            Coordinates for the ``var`` axis, which is indexed by the ``soma_joinid`` value.
            May be an ``int``, a list of ``int``, or a slice. The default, ``None``, selects all.
        column_names:
            Columns to fetch.
        modality:
            Which modality to query, can be one of ``"census_data"`` or ``"census_spatial_sequencing"``.
            Defaults to ``"census_data"``.

    Returns:
        A :class:`pandas.DataFrame` object containing metadata for the queried slice.
    """
    return _get_axis_metadata(
        census,
        "var",
        organism,
        value_filter=value_filter,
        coords=coords,
        column_names=column_names,
        modality=modality,
    )
