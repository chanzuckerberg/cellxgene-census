from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import tiledbsoma as soma

import cellxgene_census
from cellxgene_census.experimental.pp import get_highly_variable_genes, highly_variable_genes


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize(
    "experiment_name,obs_value_filter",
    [
        ("mus_musculus", 'is_primary_data == True and tissue_general == "skin of body"'),
        pytest.param(
            "mus_musculus",
            'is_primary_data == True and tissue_general in ["heart", "lung"]',
            marks=pytest.mark.expensive,
        ),
        pytest.param("mus_musculus", 'is_primary_data == True and assay == "Smart-seq"', marks=pytest.mark.expensive),
    ],
)
@pytest.mark.parametrize("n_top_genes", (5, 500))
@pytest.mark.parametrize(
    "batch_key",
    (
        None,
        "dataset_id",
        pytest.param("sex", marks=pytest.mark.expensive),
        pytest.param("tissue_general", marks=pytest.mark.expensive),
        pytest.param("assay", marks=pytest.mark.expensive),
        pytest.param("sex", marks=pytest.mark.expensive),
    ),
)
@pytest.mark.parametrize("span", (None, 0.5))
def test_hvg_vs_scanpy(
    n_top_genes: int,
    obs_value_filter: str,
    experiment_name: str,
    batch_key: Optional[str],
    span: float,
    small_mem_context: soma.SOMATileDBContext,
) -> None:
    """Compare results with ScanPy on a couple of simple tests."""

    kwargs: Dict[str, Union[int, str, float, None]] = dict(
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        flavor="seurat_v3",
    )
    if span is not None:
        kwargs["span"] = span

    with cellxgene_census.open_soma(census_version="stable", context=small_mem_context) as census:
        # Get the highly variable genes
        with census["census_data"][experiment_name].axis_query(
            measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
        ) as query:
            hvg = highly_variable_genes(query, **kwargs)  # type: ignore[arg-type]
            adata = query.to_anndata(X_name="raw")

    scanpy_hvg = sc.pp.highly_variable_genes(adata, inplace=False, **kwargs)
    scanpy_hvg.index.name = "soma_joinid"
    scanpy_hvg.index = scanpy_hvg.index.astype(int)
    assert len(scanpy_hvg) == len(hvg)
    assert all(scanpy_hvg.keys() == hvg.keys())

    assert (hvg.index == scanpy_hvg.index).all()
    assert np.allclose(hvg.means.to_numpy(), scanpy_hvg.means.to_numpy(), atol=1e-5, rtol=1e-2, equal_nan=True)
    assert np.allclose(hvg.variances.to_numpy(), scanpy_hvg.variances.to_numpy(), atol=1e-5, rtol=1e-2, equal_nan=True)
    if "highly_variable_nbatches" in scanpy_hvg.keys() or "highly_variable_nbatches" in hvg.keys():
        assert (hvg.highly_variable_nbatches == scanpy_hvg.highly_variable_nbatches).all()
    assert np.allclose(
        hvg.variances_norm.to_numpy(), scanpy_hvg.variances_norm.to_numpy(), atol=1e-5, rtol=1e-2, equal_nan=True
    )
    assert (hvg.highly_variable == scanpy_hvg.highly_variable).all()

    # Online calculation of normalized variance  will differ slightly from ScanPy's calculation,
    # so look for rank of HVGs to be close, but not identical.  Don't worry about the non-HVGs
    # (which will differ more as you get to the long tail).  This test just looks for the average
    # rank distance to be very small.
    assert (
        (
            scanpy_hvg[scanpy_hvg.highly_variable].highly_variable_rank - hvg[hvg.highly_variable].highly_variable_rank
        ).sum()
        / n_top_genes
    ) < 0.01


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize(
    "experiment_name,organism,obs_value_filter,batch_key",
    [
        ("mus_musculus", "Mus musculus", 'tissue_general == "liver" and is_primary_data == True', None),
        ("mus_musculus", "Mus musculus", 'is_primary_data == True and tissue_general == "heart"', "cell_type"),
        pytest.param(
            "mus_musculus", "Mus musculus", "is_primary_data == True", "dataset_id", marks=pytest.mark.expensive
        ),
        pytest.param(
            "homo_sapiens", "Homo sapiens", "is_primary_data == True", "dataset_id", marks=pytest.mark.expensive
        ),
    ],
)
def test_get_highly_variable_genes(
    organism: str,
    experiment_name: str,
    obs_value_filter: str,
    batch_key: str,
    small_mem_context: soma.SOMATileDBContext,
) -> None:
    with cellxgene_census.open_soma(census_version="stable", context=small_mem_context) as census:
        hvg = get_highly_variable_genes(
            census, organism=organism, obs_value_filter=obs_value_filter, n_top_genes=1000, batch_key=batch_key
        )
        n_vars = census["census_data"][experiment_name].ms["RNA"].var.count

    assert isinstance(hvg, pd.DataFrame)
    assert len(hvg) == n_vars
    assert len(hvg[hvg.highly_variable]) == 1000


@pytest.mark.experimental
def test_hvg_error_cases(small_mem_context: soma.SOMATileDBContext) -> None:
    with cellxgene_census.open_soma(census_version="stable", context=small_mem_context) as census:
        with census["census_data"]["mus_musculus"].axis_query(measurement_name="RNA") as query:
            # Only flavor="seurat_v3" is supported
            with pytest.raises(ValueError):
                highly_variable_genes(query, flavor="oopsie")  # type: ignore[arg-type]


@pytest.mark.experimental
@pytest.mark.live_corpus
def test_max_loess_jitter_error(small_mem_context: soma.SOMATileDBContext) -> None:
    with cellxgene_census.open_soma(census_version="stable", context=small_mem_context) as census:
        with pytest.raises(ValueError):
            get_highly_variable_genes(
                census,
                organism="mus_musculus",
                obs_value_filter='is_primary_data == True and tissue_general == "heart"',
                batch_key="cell_type",
                max_loess_jitter=0.0,
            )
