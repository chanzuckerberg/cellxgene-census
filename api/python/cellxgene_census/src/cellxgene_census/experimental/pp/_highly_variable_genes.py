from __future__ import annotations

import os
from concurrent import futures
from typing import Any, Callable, Sequence, cast

import numpy as np
import pandas as pd
import tiledbsoma as soma
from somacore.options import SparseDFCoord
from typing_extensions import Literal

from ..._experiment import _get_experiment
from ..util._eager_iter import _EagerIterator
from ._online import CountsAccumulator, MeanVarianceAccumulator

"""
Acknowledgements: ScanPy highly variable genes implementation (scanpy.pp.highly_variable_genes), in turn
based upon the original implementation in Seurat V3.

Ref:
* https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html#scanpy.pp.highly_variable_genes
* github.com/scverse/scanpy

Notes:
* Occasionally, skmis.loess will fail with a ValueError similar to
    `ValueError: b'reciprocal condition number  2.0467e-15\n'`.
  or
    `ValueError: b'There are other near singularities as well. 0.090619\n'`
  This is likely caused by an excess of all-zero valued counts in a given batch or
  other low-entropy data. Ref:
    * https://github.com/scverse/scanpy/issues/1504
    * https://github.com/has2k1/scikit-misc/issues/9
    * https://discourse.scverse.org/t/error-in-highly-variable-gene-selection/276/9

  It seems possible to work around by retrying failures with addition of noise/jitter.
"""


def _get_batch_index(
    query: soma.ExperimentAxisQuery,
    batch_key: str | Sequence[str],
    batch_key_func: Callable[..., Any] | None = None,
) -> pd.Series[Any]:
    """Return categorical series representing the batch key, with codes that index the key."""
    if isinstance(batch_key, str):
        batch_key = [batch_key]
    batch_key = list(batch_key)
    assert isinstance(batch_key, list) and len(batch_key) > 0

    obs: pd.DataFrame = (
        query.obs(column_names=["soma_joinid"] + batch_key).concat().to_pandas().set_index("soma_joinid")[batch_key]
    )

    batch_series: pd.Series[Any]
    if batch_key_func is not None:
        # apply user lambda
        batch_series = obs.apply(batch_key_func, axis=1, result_type="reduce")
    elif len(batch_key) > 1:
        # if multiple keys, stringify and concat
        obs = obs.astype(str)
        batch_series = obs[cast(str, batch_key[0])]
        batch_series = batch_series.str.cat(obs[batch_key[1:]])
    else:
        # if a single key, just use it
        batch_series = obs[cast(str, batch_key[0])]

    batch_series = batch_series.astype("category")
    return batch_series


def _highly_variable_genes_seurat_v3(
    query: soma.ExperimentAxisQuery,
    batch_key: str | Sequence[str] | None = None,
    n_top_genes: int = 1_000,
    layer: str = "raw",
    span: float = 0.3,
    max_loess_jitter: float = 1e-6,
    batch_key_func: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    try:
        import skmisc.loess
    except ImportError as e:
        raise ImportError("Please install skmisc package via `pip install --user scikit-misc") from e

    batch_indexer = None
    if batch_key is not None:
        batch_index = _get_batch_index(query, batch_key, batch_key_func)
        n_batches = len(batch_index.cat.categories)
        n_samples = batch_index.value_counts().loc[batch_index.cat.categories.to_numpy()].to_numpy()
        if n_batches > 1:
            batch_indexer = soma.tiledbsoma_build_index(
                batch_index.index.to_numpy(), context=query.experiment.context
            ).get_indexer
            batch_codes = batch_index.cat.codes.to_numpy().astype(np.int64)
    else:
        n_batches = 1
        n_samples = np.array([query.n_obs], dtype=np.int64)

    assert n_batches == len(n_samples)
    assert query.n_obs == n_samples.sum()
    assert all(n_samples > 0)
    assert (n_batches > 1) == bool(batch_indexer)

    max_workers = (os.cpu_count() or 4) + 2
    var_indexer = query.indexer

    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        mvn = MeanVarianceAccumulator(n_batches, n_samples, query.n_vars)
        for arrow_tbl in _EagerIterator(query.X(layer).tables(), pool=pool):
            data = arrow_tbl["soma_data"].to_numpy()
            if batch_indexer:
                _batch_take_at_future = pool.submit(batch_indexer, arrow_tbl["soma_dim_0"])
                var_dim = var_indexer.by_var(arrow_tbl["soma_dim_1"])
                _batch_vec = batch_codes[_batch_take_at_future.result()]
                mvn.update(var_dim, data, _batch_vec)
            else:
                var_dim = var_indexer.by_var(arrow_tbl["soma_dim_1"])
                mvn.update(var_dim, data)

        batches_u, batches_var, all_u, all_var = mvn.finalize()
        del mvn

    var_df = pd.DataFrame(
        index=pd.Index(data=query.var_joinids(), name="soma_joinid"),
        data={
            "means": all_u,
            "variances": all_var,
        },
    )

    # Calculate per-batch clip_val and reg_std
    estimated_variances = np.empty((query.n_vars,), dtype=np.float64)
    reg_std = np.zeros((n_batches, query.n_vars), dtype=np.float64)
    clip_val = np.zeros((n_batches, query.n_vars), dtype=np.float64)
    for batch in range(n_batches):
        estimated_variances.fill(0)
        u = batches_u[batch]
        v = batches_var[batch]
        N = n_samples[batch]

        not_const = v > 0
        if N == 1 or not not_const.any():
            reg_std[batch].fill(1)
            clip_val[batch, :] = u
            continue

        y = np.log10(v[not_const])
        x = np.log10(u[not_const])

        jitter_magnitude: float = 0
        while True:
            try:
                # Attempt to resolve low entropy loess errors by adding jitter and retrying
                # See: https://github.com/has2k1/scikit-misc/issues/9
                if jitter_magnitude != 0:
                    _x = x + np.random.default_rng().uniform(-jitter_magnitude, jitter_magnitude, x.shape[0])
                else:
                    _x = x

                model = skmisc.loess.loess(_x, y, span=span, degree=2)
                model.fit()
                estimated_variances[not_const] = model.outputs.fitted_values
                break

            except ValueError:
                jitter_magnitude = 1e-18 if jitter_magnitude == 0 else jitter_magnitude * 10.0
                if jitter_magnitude < max_loess_jitter:
                    continue
                raise

        reg_std[batch] = np.sqrt(10**estimated_variances)
        vmax = np.sqrt(N)
        clip_val[batch] = reg_std[batch] * vmax + u

    del estimated_variances

    # Read counts again, clip and save sum of counts and sum of counts squared
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        acc = CountsAccumulator(n_batches, query.n_vars, clip_val)
        for arrow_tbl in _EagerIterator(query.X(layer).tables(), pool=pool):
            data = arrow_tbl["soma_data"].to_numpy()
            if batch_indexer:
                _batch_take_at_future = pool.submit(batch_indexer, arrow_tbl["soma_dim_0"])
                var_dim = var_indexer.by_var(arrow_tbl["soma_dim_1"])
                _batch_vec = batch_codes[_batch_take_at_future.result()]
                acc.update(var_dim, data, _batch_vec)
            else:
                var_dim = var_indexer.by_var(arrow_tbl["soma_dim_1"])
                acc.update(var_dim, data)

        counts_sum, squared_counts_sum = acc.finalize()
        # Don't raise python errors for 0/0, etc, just generate Inf/NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_gene_vars = (1 / ((n_samples - 1) * np.square(reg_std.T))).T * (
                (n_samples * np.square(batches_u.T)).T + squared_counts_sum - 2 * counts_sum * batches_u
            )
            norm_gene_vars[np.isnan(norm_gene_vars)] = 0
        del acc, counts_sum, squared_counts_sum

    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum((ranked_norm_gene_vars < n_top_genes).astype(int), axis=0)
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)  # type: ignore[no-untyped-call]
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)  # type: ignore[no-untyped-call]

    var_df = var_df.assign(
        highly_variable_nbatches=pd.Series(num_batches_high_var, index=var_df.index),
        highly_variable_rank=pd.Series(median_ranked, index=var_df.index),
        variances_norm=pd.Series(np.mean(norm_gene_vars, axis=0), index=var_df.index),
    )

    sorted_index = (
        var_df[["highly_variable_rank", "highly_variable_nbatches"]]
        .sort_values(
            ["highly_variable_rank", "highly_variable_nbatches"],
            ascending=[True, False],
            na_position="last",
        )
        .index
    )
    var_df["highly_variable"] = False
    var_df.loc[sorted_index[: int(n_top_genes)], "highly_variable"] = True
    if batch_key is None:
        var_df = var_df.drop(columns=["highly_variable_nbatches"])
    return var_df


def highly_variable_genes(
    query: soma.ExperimentAxisQuery,
    n_top_genes: int = 1_000,
    layer: str = "raw",
    flavor: Literal["seurat_v3"] = "seurat_v3",
    span: float = 0.3,
    batch_key: str | Sequence[str] | None = None,
    max_loess_jitter: float = 1e-6,
    batch_key_func: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Identify and annotate highly variable genes contained in the query results.
    The API is modelled on ScanPy `scanpy.pp.highly_variable_genes` API.
    Results returned will mimic ScanPy results. The only `flavor` available
    is the Seurat V3 method, which assumes count data in the X layer.

    See
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html#scanpy.pp.highly_variable_genes
    for more information on this method.

    Args:
        query:
            A :class:`tiledbsoma.ExperimentAxisQuery`, specifying the ``obs``/``var`` selection over which genes are
            annotated.
        n_top_genes:
            Number of genes to rank.
        layer:
            X layer used, e.g., ``"raw"``.
        flavor:
            Method used to annotate genes. Must be ``"seurat_v3"``.
        span:
            If ``flavor="seurat_v3"``, the fraction of obs/cells used to estimate the LOESS variance model fit.
        batch_key:
            If specified, gene selection will be done by batch and combined. Specify the obs column name, or list of
            column names, identifying the batches. If not specified, all gene selection is done as a single batch.
            If multiple batch keys are specified, and no batch_key_func is specified, the batch key will be generated by
            converting values to string and concatenating them.
        max_lowess_jitter:
            The maximum jitter to add to data in case of LOESS failure (can occur when dataset has low entry counts.)
        batch_key_func:
            Optional function to create a user-defined batch key. Function will be called once per row in the obs
            dataframe. Function will receive a single argument: a :class:`pandas.Series` containing values specified in
            the``batch_key`` argument.

    Returns:
        A :class:`pandas.DataFrame` containing annotations for all ``var`` values specified by the ``query`` argument.
        Annotations are identical to those produced by :func:`scanpy.pp.highly_variable_genes`.

    Raises:
        ValueError: if the flavor parameter is not ``"seurat_v3"``.


    Examples:
        Fetch :class:`pandas.DataFrame` containing var annotations for the query selection, using ``"dataset_id"`` as
        ``batch_key``.

        >>> hvg = highly_variable_genes(query, batch_key="dataset_id")

        Fetch highly variable genes, using the concatenation of ``"dataset_id"`` and ``"donor_id"`` as ``batch_key``:

        >>> hvg = highly_variable_genes(query, batch_key=["dataset_id", "donor_id"])

        Fetch highly variable genes, with a user-defined ``batch_key_func``:

        >>> hvg = highly_variable_genes(
                query,
                batch_key="donor_id",
                batch_key_func=lambda s: return "batch0" if s.donor_id == "99" else "batch1"
            )

    Lifecycle:
        experimental
    """
    if flavor != "seurat_v3":
        raise ValueError('`flavor` must be "seurat_v3"')

    return _highly_variable_genes_seurat_v3(
        query,
        n_top_genes=n_top_genes,
        layer=layer,
        span=span,
        batch_key=batch_key,
        batch_key_func=batch_key_func,
        max_loess_jitter=max_loess_jitter,
    )


def get_highly_variable_genes(
    census: soma.Collection,
    organism: str,
    measurement_name: str = "RNA",
    X_name: str = "raw",
    obs_value_filter: str | None = None,
    obs_coords: SparseDFCoord | None = None,
    var_value_filter: str | None = None,
    var_coords: SparseDFCoord | None = None,
    n_top_genes: int = 1_000,
    flavor: Literal["seurat_v3"] = "seurat_v3",
    span: float = 0.3,
    batch_key: str | Sequence[str] | None = None,
    max_loess_jitter: float = 1e-6,
    batch_key_func: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Convience wrapper around :class:`tiledbsoma.Experiment` query and
    :func:`cellxgene_census.experimental.pp.highly_variable_genes` function, to build and execute a query, and annotate
    the query result genes (``var`` dataframe) based upon variability.

    Args:
        census:
            The Census object, usually returned by :func:`open_soma`.
        organism:
            The organism to query, usually one of ``"Homo sapiens"`` or ``"Mus musculus"``.
        measurement_name:
            The measurement object to query. Defaults to ``"RNA"``.
        X_name:
            The ``X`` layer to query. Defaults to ``"raw"``.
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
        n_top_genes:
            Number of genes to rank.
        flavor:
            Method used to annotate genes. Must be ``"seurat_v3"``.
        span:
            If ``flavor="seurat_v3"``, the fraction of obs/cells used to estimate the LOESS variance model fit.
        batch_key:
            If specified, gene selection will be done by batch and combined. Specify the obs column name, or list of
            column names, identifying the batches. If not specified, all gene selection is done as a single batch.
            If multiple batch keys are specified, and no batch_key_func is specified, the batch key will be generated by
            converting values to string and concatenating them.
        max_lowess_jitter:
            The maximum jitter to add to data in case of LOESS failure (can occur when dataset has low entry counts.)
        batch_key_func:
            Optional function to create a user-defined batch key. Function will be called once per row in the obs
            dataframe. Function will receive a single argument: a :class:`pandas.Series` containing values specified in
            the ``batch_key`` argument.

    Returns:
        :class:`pandas.DataFrame` containing annotations for all ``var`` values specified by the query.

    Raises:
        ValueError: if the flavor paramater is not ``"seurat_v3"``.

    See Also:
        :func:`cellxgene_census.experimental.pp.highly_variable_genes`

    Examples:
        Fetch a :class:`pandas.DataFrame` containing var annotations for a subset of the cells matching the
        ``obs_value_filter``:

        >>> hvg = get_highly_variable_genes(
                census,
                organism="Mus musculus",
                obs_value_filter="is_primary_data == True and tissue_general == 'lung'",
                n_top_genes = 500
            )

        Fetch an :class:`anndata.AnnData` with top 500 genes:

        >>> with cellxgene_census.open_soma(census_version="stable") as census:
                organism = "mus_musculus"
                obs_value_filter = "is_primary_data == True and tissue_general == 'lung'"
                # Get the highly variable genes
                hvg = cellxgene_census.experimental.pp.get_highly_variable_genes(
                    census,
                    organism=organism,
                    obs_value_filter=obs_value_filter,
                    n_top_genes = 500
                )
                # Fetch AnnData - all cells matching obs_value_filter, just the HVGs
                hvg_soma_ids = hvg[hvg.highly_variable].index.values
                adata = cellxgene_census.get_anndata(
                    census, organism=organism, obs_value_filter=obs_value_filter, var_coords=hvg_soma_ids
                )

    Lifecycle:
        experimental

    """
    exp = _get_experiment(census, organism)
    obs_coords = (slice(None),) if obs_coords is None else (obs_coords,)
    var_coords = (slice(None),) if var_coords is None else (var_coords,)
    with exp.axis_query(
        measurement_name,
        obs_query=soma.AxisQuery(value_filter=obs_value_filter, coords=obs_coords),
        var_query=soma.AxisQuery(value_filter=var_value_filter, coords=var_coords),
    ) as query:
        return highly_variable_genes(
            query,
            n_top_genes=n_top_genes,
            layer=X_name,
            flavor=flavor,
            span=span,
            batch_key=batch_key,
            batch_key_func=batch_key_func,
            max_loess_jitter=max_loess_jitter,
        )
