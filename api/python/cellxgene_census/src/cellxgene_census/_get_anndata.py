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

from ._experiment import _get_experiment
from ._util import _extract_census_version

CENSUS_EMBEDDINGS_LOCATION_BASE_URI = "s3://cellxgene-contrib-public/contrib/cell-census/soma"


def get_anndata(
    census: soma.Collection,
    organism: str,
    measurement_name: str = "RNA",
    X_name: str = "raw",
    X_layers: Optional[Sequence[str]] = (),
    obsm_layers: Optional[Sequence[str]] = (),
    varm_layers: Optional[Sequence[str]] = (),
    obs_value_filter: Optional[str] = None,
    obs_coords: Optional[SparseDFCoord] = None,
    var_value_filter: Optional[str] = None,
    var_coords: Optional[SparseDFCoord] = None,
    column_names: Optional[soma.AxisColumnNames] = None,
    add_obs_embeddings: Optional[Sequence[str]] = (),
    add_var_embeddings: Optional[Sequence[str]] = (),
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
        varm_layers:
            Additional varm layers to read and return in the ``varm`` slot.
        add_obs_embeddings:
            Embeddings to be returned as part of the ``obsm`` slot.
            Use :func:`get_all_available_embeddings` to retrieve available embeddings
            for this Census version and organism.
        add_var_embeddings:
            Embeddings to be returned as part of the ``varm`` slot.
            Use :func:`get_all_available_embeddings` to retrieve available embeddings
            for this Census version and organism.

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
        adata = query.to_anndata(
            X_name=X_name,
            column_names=column_names,
            X_layers=X_layers,
            obsm_layers=obsm_layers,
            varm_layers=varm_layers,
        )

        # If add_obs_embeddings or add_var_embeddings are defined, inject them in the appropriate slot
        if add_obs_embeddings is not None:
            obs_soma_joinids = query.obs_joinids()
            from cellxgene_census.experimental import get_embedding, get_embedding_metadata_by_name

            census_version = _extract_census_version(census)
            for emb in add_obs_embeddings:
                emb_metadata = get_embedding_metadata_by_name(emb, organism, census_version, "obs_embedding")
                uri = f"{CENSUS_EMBEDDINGS_LOCATION_BASE_URI}/{census_version}/{emb_metadata['id']}"
                embedding = get_embedding(census_version, uri, obs_soma_joinids)
                adata.obsm[emb] = embedding

        return adata
