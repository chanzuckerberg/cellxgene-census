import pathlib
from typing import List

import anndata as ad
import numpy as np
import pytest
from cellxgene_census_builder.build_soma.anndata import make_anndata_cell_filter2, open_anndata2
from cellxgene_census_builder.build_soma.datasets import Dataset

from ..conftest import ORGANISMS


def test_open_anndata(datasets: List[Dataset]) -> None:
    """
    `open_anndata` should open the h5ads for each of the dataset in the argument,
    and yield both the dataset and the corresponding AnnData object.
    This test does not involve additional filtering steps.
    The `datasets` used here have no raw layer.
    """
    result = [(d, open_anndata2(".", d)) for d in datasets]
    assert len(result) == len(datasets)
    for i, (dataset, anndata_obj) in enumerate(result):
        assert dataset == datasets[i]
        opened_anndata = ad.read_h5ad(dataset.dataset_h5ad_path)
        assert opened_anndata.obs.equals(anndata_obj.obs)
        assert opened_anndata.var.equals(anndata_obj.var)
        assert np.array_equal(opened_anndata.X.todense(), anndata_obj.X.todense())


def test_open_anndata_filters_out_datasets_with_mixed_feature_reference(
    datasets_with_mixed_feature_reference: List[Dataset],
) -> None:
    """
    Datasets with a "mixed" feature_reference will not be included by the filter pipeline
    """
    ad_filter = make_anndata_cell_filter2({})
    result = [ad_filter(open_anndata2(".", d)) for d in datasets_with_mixed_feature_reference]
    assert all(len(ad) == 0 for ad in result)


def test_open_anndata_filters_out_wrong_schema_version_datasets(
    caplog: pytest.LogCaptureFixture,
    datasets_with_incorrect_schema_version: List[Dataset],
) -> None:
    """
    Datasets with a schema version different from `CXG_SCHEMA_VERSION` will not be included by `open_anndata`
    """
    for dataset in datasets_with_incorrect_schema_version:
        with pytest.raises(ValueError, match="incorrect CxG schema version"):
            _ = open_anndata2(".", dataset)


def test_make_anndata_cell_filter(tmp_path: pathlib.Path, h5ad_simple: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_simple)
    adata_simple = open_anndata2(tmp_path.as_posix(), dataset)

    func = make_anndata_cell_filter2({})
    filtered_adata_simple = func(adata_simple)

    assert adata_simple.var.equals(filtered_adata_simple.var)
    assert adata_simple.obs.equals(filtered_adata_simple.obs)
    assert np.array_equal(adata_simple.X.todense(), filtered_adata_simple.X.todense())


def test_make_anndata_cell_filter_filters_out_organoids_cell_culture(
    tmp_path: pathlib.Path,
    h5ad_with_organoids_and_cell_culture: str,
) -> None:
    dataset = Dataset(
        dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_with_organoids_and_cell_culture
    )
    adata_with_organoids_and_cell_culture = open_anndata2(tmp_path.as_posix(), dataset)

    func = make_anndata_cell_filter2({})
    filtered_adata_with_organoids_and_cell_culture = func(adata_with_organoids_and_cell_culture)

    assert adata_with_organoids_and_cell_culture.var.equals(filtered_adata_with_organoids_and_cell_culture.var)
    assert filtered_adata_with_organoids_and_cell_culture.obs.shape[0] == 2


def test_make_anndata_cell_filter_organism(tmp_path: pathlib.Path, h5ad_with_organism: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_with_organism)
    adata_with_organism = open_anndata2(tmp_path.as_posix(), dataset)

    func = make_anndata_cell_filter2({"organism_ontology_term_id": ORGANISMS[0].organism_ontology_term_id})
    filtered_adata_with_organism = func(adata_with_organism)

    assert adata_with_organism.var.equals(filtered_adata_with_organism.var)
    assert filtered_adata_with_organism.obs.shape[0] == 3


def test_make_anndata_cell_filter_feature_biotype_gene(tmp_path: pathlib.Path, h5ad_with_feature_biotype: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_with_feature_biotype)
    adata_with_feature_biotype = open_anndata2(tmp_path.as_posix(), dataset)

    func = make_anndata_cell_filter2({})
    filtered_adata_with_feature_biotype = func(adata_with_feature_biotype)

    assert adata_with_feature_biotype.obs.equals(filtered_adata_with_feature_biotype.obs)
    assert filtered_adata_with_feature_biotype.var.shape[0] == 3


def test_make_anndata_cell_filter_assay(tmp_path: pathlib.Path, h5ad_with_assays: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_with_assays)
    adata_with_assays = open_anndata2(tmp_path.as_posix(), dataset, include_filter_columns=True)

    func = make_anndata_cell_filter2({"assay_ontology_term_ids": ["EFO:1234", "EFO:1235"]})
    filtered_adata_with_assays = func(adata_with_assays)

    assert filtered_adata_with_assays.obs.shape[0] == 2
    assert list(filtered_adata_with_assays.obs.index) == [0, 2]
