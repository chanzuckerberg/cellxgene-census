from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import tiledbsoma as soma

import cellxgene_census
from cellxgene_census.experimental.pp import (
    get_highly_variable_genes,
    highly_variable_genes,
)


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize(
    "experiment_name,obs_value_filter",
    [
        (
            "mus_musculus",
            'is_primary_data == True and tissue_general == "liver"',
        ),
        pytest.param(
            "mus_musculus",
            'is_primary_data == True and tissue_general == "skin of body"',
            marks=pytest.mark.expensive,
        ),
        pytest.param(
            "mus_musculus",
            'is_primary_data == True and tissue_general in ["heart", "lung"]',
            marks=pytest.mark.expensive,
        ),
        pytest.param(
            "mus_musculus",
            'is_primary_data == True and assay == "Smart-seq2"',
            marks=pytest.mark.expensive,
        ),
    ],
)
@pytest.mark.parametrize("n_top_genes", (50, 500))
@pytest.mark.parametrize(
    "batch_key",
    (
        None,
        "dataset_id",
        ["suspension_type", "assay_ontology_term_id"],
        pytest.param(
            ("suspension_type", "assay_ontology_term_id"),
            marks=pytest.mark.expensive,
        ),
    ),
)
@pytest.mark.parametrize(
    "span",
    (
        pytest.param(None, marks=pytest.mark.expensive),
        0.5,
    ),
)
@pytest.mark.parametrize(
    "version",
    (
        "latest",
        pytest.param("stable", marks=pytest.mark.expensive),
    ),
)
def test_hvg_vs_scanpy(
    n_top_genes: int,
    obs_value_filter: str,
    version: str,
    experiment_name: str,
    batch_key: str | tuple[str] | list[str] | None,
    span: float,
    small_mem_context: soma.SOMATileDBContext,
) -> None:
    """Compare results with ScanPy on a couple of simple tests."""

    kwargs: dict[str, Any] = {
        "n_top_genes": n_top_genes,
        "batch_key": batch_key,
        "flavor": "seurat_v3",
    }
    if span is not None:
        kwargs["span"] = span

    with cellxgene_census.open_soma(census_version=version, context=small_mem_context) as census:
        # Get the highly variable genes
        with census["census_data"][experiment_name].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=obs_value_filter),
        ) as query:
            hvg = highly_variable_genes(query, **kwargs)
            adata = query.to_anndata(X_name="raw")

    if isinstance(batch_key, list) or isinstance(batch_key, tuple):
        # ScanPy only accepts a single column for a batch key, so create it and use it
        assert "the_batch_key" not in adata.obs.columns
        adata.obs["the_batch_key"] = (
            adata.obs[list(batch_key)].astype(str)[batch_key[0]].str.cat(adata.obs[list(batch_key[1:])])
        )
        kwargs["batch_key"] = "the_batch_key"

    try:
        scanpy_hvg = sc.pp.highly_variable_genes(adata, inplace=False, **kwargs)
    except (ZeroDivisionError, ValueError):
        # There are test cases where ScanPy will fail, rendering this "compare vs scanpy"
        # test moot. The known cases involve overly partitioned data that results in batches
        # with a very small number of samples (which manifest as a divide by zero error).
        # In these known cases, go ahead and perform the HVG (above), but skip the compare
        # assertions below.
        pytest.skip("ScanPy generated an error, likely due to batches with 1 sample")

    scanpy_hvg.index.name = "soma_joinid"
    scanpy_hvg.index = scanpy_hvg.index.astype(int)
    assert len(scanpy_hvg) == len(hvg)
    assert all(scanpy_hvg.keys() == hvg.keys())

    assert (hvg.index == scanpy_hvg.index).all()
    assert np.allclose(
        hvg.means.to_numpy(),
        scanpy_hvg.means.to_numpy(),
        atol=1e-5,
        rtol=1e-2,
        equal_nan=True,
    )
    assert np.allclose(
        hvg.variances.to_numpy(),
        scanpy_hvg.variances.to_numpy(),
        atol=1e-5,
        rtol=1e-2,
        equal_nan=True,
    )
    assert np.allclose(
        hvg.variances_norm.to_numpy(),
        scanpy_hvg.variances_norm.to_numpy(),
        atol=1e-5,
        rtol=1e-2,
        equal_nan=True,
    )

    # Online calculation of normalized variance will differ slightly from ScanPy's calculation,
    # so look for rank of HVGs to be close, but not identical.  Don't worry about the non-HVGs
    # (which will differ more as you get to the long tail).  This test just looks for the average
    # rank distance to be small.
    assert (
        (scanpy_hvg[scanpy_hvg.highly_variable].highly_variable_rank - hvg[hvg.highly_variable].highly_variable_rank)
        .abs()
        .sum()
        / n_top_genes
    ) < 0.05

    # Ranking will also have some noise, so check that ranking is close in the highly variable subset
    scanpy_rank = scanpy_hvg.highly_variable_rank.copy()
    hvg_rank = hvg.highly_variable_rank.copy()
    hvg_rank[pd.isna(hvg_rank)] = n_top_genes
    scanpy_rank[pd.isna(scanpy_rank)] = n_top_genes
    rank_diff = (hvg_rank - scanpy_rank)[hvg.highly_variable]
    # +/- 5 in ranking, choosen arbitrarily
    assert rank_diff.min() >= -5 and rank_diff.max() <= 5

    if "highly_variable_nbatches" in scanpy_hvg.keys() or "highly_variable_nbatches" in hvg.keys():
        # Also subject to noise, so look for "close" match
        nbatches_diff = hvg.highly_variable_nbatches - scanpy_hvg.highly_variable_nbatches
        assert nbatches_diff.min() >= -2 and nbatches_diff.max() <= 2

    assert (hvg.highly_variable == scanpy_hvg.highly_variable).all()


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize(
    "experiment_name,organism,obs_value_filter,batch_key,obs_coords",
    [
        (
            "mus_musculus",
            "Mus musculus",
            'tissue_general == "liver" and is_primary_data == True',
            None,
            None,
        ),
        (
            "mus_musculus",
            "Mus musculus",
            'is_primary_data == True and tissue_general == "heart"',
            "dataset_id",
            None,
        ),
        (
            "mus_musculus",
            "Mus musculus",
            'is_primary_data == True and tissue_general == "heart"',
            ["suspension_type", "assay_ontology_term_id"],
            None,
        ),
        pytest.param(
            "mus_musculus",
            "Mus musculus",
            "is_primary_data == True",
            "dataset_id",
            slice(750_000, 1_000_000),
            marks=pytest.mark.expensive,
        ),
        pytest.param(
            "homo_sapiens",
            "Homo sapiens",
            "is_primary_data == True",
            "dataset_id",
            slice(1_000_000, 4_000_000),
            marks=pytest.mark.expensive,
        ),
    ],
)
def test_get_highly_variable_genes(
    organism: str,
    experiment_name: str,
    obs_value_filter: str,
    batch_key: str,
    small_mem_context: soma.SOMATileDBContext,
    obs_coords: slice | None,
) -> None:
    with cellxgene_census.open_soma(census_version="stable", context=small_mem_context) as census:
        hvg = get_highly_variable_genes(
            census,
            organism=organism,
            obs_value_filter=obs_value_filter,
            n_top_genes=1000,
            batch_key=batch_key,
            obs_coords=obs_coords,
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


@pytest.mark.experimental
@pytest.mark.live_corpus
@pytest.mark.parametrize(
    "batch_key",
    [
        None,
        "suspension_type",
        ["assay_ontology_term_id", "suspension_type"],
        ["dataset_id", "assay_ontology_term_id", "suspension_type", "donor_id"],
    ],
)
def test_hvg_user_defined_batch_key_func(
    small_mem_context: soma.SOMATileDBContext,
    batch_key: str | list[str] | None,
) -> None:
    if batch_key is None:

        def batch_key_func(srs: pd.Series[Any]) -> str:
            raise AssertionError("should never be called without a batch key")

    else:
        if isinstance(batch_key, str):
            keys = set([batch_key])  # noqa: C405
        else:
            keys = set(batch_key)

        def batch_key_func(srs: pd.Series[Any]) -> str:
            assert set(srs.keys()) == keys
            return "batch0"

    with cellxgene_census.open_soma(census_version="latest", context=small_mem_context) as census:
        # Get the highly variable genes
        with census["census_data"]["mus_musculus"].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(coords=(slice(75000),)),
        ) as query:
            hvg = highly_variable_genes(
                query,
                batch_key=batch_key,
                batch_key_func=batch_key_func,
                n_top_genes=100,
            )

            assert len(hvg[hvg.highly_variable]) == 100
