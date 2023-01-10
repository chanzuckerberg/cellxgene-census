import numpy as np
import pandas as pd
import tiledbsoma as soma

from .meanvar import OnlineMatrixMeanVariance


def highly_variable_genes(query: soma.experiment_query.ExperimentAxisQuery, n_top_genes: int = 10) -> pd.DataFrame:
    """
    Acknowledgements: scanpy highly variable genes implementation, github.com/scverse/scanpy
    """
    use_prefetch = True

    try:
        import skmisc.loess
    except ImportError:
        raise ImportError("Please install skmisc package via `pip install --user scikit-misc")

    indexer = query.get_indexer()
    mvn = OnlineMatrixMeanVariance(query.n_obs, query.n_vars)
    for arrow_tbl in query.X("raw", prefetch=use_prefetch):
        var_dim = indexer.var_index(arrow_tbl["soma_dim_1"])
        data = arrow_tbl["soma_data"].to_numpy()
        mvn.update(var_dim, data)

    u, v = mvn.finalize()
    var_df = pd.DataFrame(
        index=pd.Index(data=query.var_joinids(), name="soma_joinid"),
        data={
            "means": u,
            "variances": v,
        },
    )

    estimated_variances = np.zeros((len(var_df),), dtype=np.float64)
    not_const = v > 0
    y = np.log10(v[not_const])
    x = np.log10(u[not_const])
    model = skmisc.loess.loess(x, y, span=0.3, degree=2)
    model.fit()
    estimated_variances[not_const] = model.outputs.fitted_values
    reg_std = np.sqrt(10**estimated_variances)

    # A second pass over the data is required because the clip value
    # is determined by the first pass
    N = query.n_obs
    vmax = np.sqrt(N)
    clip_val = reg_std * vmax + u
    counts_sum = np.zeros((query.n_vars,), dtype=np.float64)  # clipped
    squared_counts_sum = np.zeros((query.n_vars,), dtype=np.float64)  # clipped
    for arrow_tbl in query.X("raw", prefetch=use_prefetch):
        var_dim = indexer.var_index(arrow_tbl["soma_dim_1"])
        data = arrow_tbl["soma_data"].to_numpy()
        # clip
        mask = data > clip_val[var_dim]
        data = data.copy()
        data[mask] = clip_val[var_dim[mask]]
        np.add.at(counts_sum, var_dim, data)
        np.add.at(squared_counts_sum, var_dim, data**2)

    norm_gene_vars = (1 / ((N - 1) * np.square(reg_std))) * (
        (N * np.square(u)) + squared_counts_sum - 2 * counts_sum * u
    )
    norm_gene_vars = norm_gene_vars.reshape(1, -1)

    # argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)

    # this is done in SelectIntegrationFeatures() in Seurat v3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum((ranked_norm_gene_vars < n_top_genes).astype(int), axis=0)
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)  # type: ignore
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)  # type: ignore

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
    var_df = var_df.drop(columns=["highly_variable_nbatches"])
    var_df.loc[sorted_index[: int(n_top_genes)], "highly_variable"] = True
    return var_df
