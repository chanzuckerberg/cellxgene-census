from typing import Dict, Optional, Union

import numpy as np
import pytest
import scanpy as sc
import tiledbsoma as soma

import cellxgene_census
from cellxgene_census.experimental.pp import highly_variable_genes


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize(
    "experiment_name,obs_value_filter",
    [
        pytest.param(
            "mus_musculus", 'is_primary_data == True and tissue_general == "lung"', marks=pytest.mark.expensive
        ),
        ("mus_musculus", 'is_primary_data == True and tissue_general == "heart"'),
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
    n_top_genes: int, obs_value_filter: str, experiment_name: str, batch_key: Optional[str], span: float
) -> None:
    """Compare results with ScanPy on a couple of simple tests."""

    kwargs: Dict[str, Union[int, str, float, None]] = dict(
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        flavor="seurat_v3",
    )
    if span is not None:
        kwargs["span"] = span

    with cellxgene_census.open_soma(census_version="stable") as census:
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
def test_hvg_error_cases() -> None:
    with cellxgene_census.open_soma(census_version="stable") as census:
        with census["census_data"]["mus_musculus"].axis_query(measurement_name="RNA") as query:
            # Only flavor="seurat_v3" is supported
            with pytest.raises(ValueError):
                highly_variable_genes(query, flavor="oopsie")  # type: ignore[arg-type]
