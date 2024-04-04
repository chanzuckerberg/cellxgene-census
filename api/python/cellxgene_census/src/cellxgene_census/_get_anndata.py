# Copyright (c) 2022, Chan Zuckerberg Initiative
#
# Licensed under the MIT License.

"""Get slice as AnnData.

Methods to retrieve slices of the census as AnnData objects.
"""

from typing import Optional, Sequence

import anndata
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
    X_layers: Optional[Sequence[str]] = (),
    obsm_layers: Optional[Sequence[str]] = (),
    obsp_layers: Optional[Sequence[str]] = (),
    varm_layers: Optional[Sequence[str]] = (),
    varp_layers: Optional[Sequence[str]] = (),
    obs_value_filter: Optional[str] = None,
    obs_coords: Optional[SparseDFCoord] = None,
    var_value_filter: Optional[str] = None,
    var_coords: Optional[SparseDFCoord] = None,
    column_names: Optional[soma.AxisColumnNames] = None,
    obs_embeddings: Optional[Sequence[str]] = (),
    var_embeddings: Optional[Sequence[str]] = (),
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
        column_names:
            Columns to fetch for ``obs`` and ``var`` dataframes.
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

    Returns:
        An :class:`anndata.AnnData` object containing the census slice.

    Lifecycle:
        experimental

    Examples:
        >>> get_anndata(census, "Mus musculus", obs_value_filter="tissue_general in ['brain', 'lung']")

        >>> get_anndata(census, "Homo sapiens", column_names={"obs": ["tissue"]})

        >>> get_anndata(census, "Homo sapiens", obs_coords=slice(0, 1000))
    """
    exp = _get_experiment(census, organism)
    obs_coords = (slice(None),) if obs_coords is None else (obs_coords,)
    var_coords = (slice(None),) if var_coords is None else (var_coords,)

    if obsm_layers and obs_embeddings and set(obsm_layers) & set(obs_embeddings):
        raise ValueError("Cannot request both `obsm_layers` and `obs_embeddings` for the same embedding name")

    if varm_layers and var_embeddings and set(varm_layers) & set(var_embeddings):
        raise ValueError("Cannot request both `varm_layers` and `var_embeddings` for the same embedding name")

    with exp.axis_query(
        measurement_name,
        obs_query=soma.AxisQuery(value_filter=obs_value_filter, coords=obs_coords),
        var_query=soma.AxisQuery(value_filter=var_value_filter, coords=var_coords),
    ) as query:
        adata = query.to_anndata(
            X_name=X_name,
            column_names=column_names,
            X_layers=X_layers,
            obsm_layers=obsm_layers,
            varm_layers=varm_layers,
            obsp_layers=obsp_layers,
            varp_layers=varp_layers,
        )

        # If obs_embeddings or var_embeddings are defined, inject them in the appropriate slot
        if obs_embeddings or var_embeddings:
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
