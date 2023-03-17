from typing import List

import anndata as ad
import numpy as np

from tools.cell_census_builder.anndata import get_cellxgene_schema_version, make_anndata_cell_filter, open_anndata
from tools.cell_census_builder.datasets import Dataset
from tools.cell_census_builder.tests.conftest import ORGANISMS


def test_open_anndata(datasets: List[Dataset]) -> None:
    """
    `open_anndata` should open the h5ads for each of the dataset in the argument,
    and yield both the dataset and the corresponding AnnData object.
    This test does not involve additional filtering steps.
    The `datasets` used here have no raw layer.
    """
    result = list(open_anndata(".", datasets))
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
    Datasets with a "mixed" feature_reference will not be included by `open_anndata`
    """
    result = list(open_anndata(".", datasets_with_mixed_feature_reference))
    assert len(result) == 0


def test_open_anndata_filters_out_wrong_schema_version_datasets(
    datasets_with_incorrect_schema_version: List[Dataset],
) -> None:
    """
    Datasets with a schema version different from `CXG_SCHEMA_VERSION` will not be included by `open_anndata`
    """
    result = list(open_anndata(".", datasets_with_incorrect_schema_version))
    assert len(result) == 0


def test_open_anndata_equalizes_raw_and_normalized(datasets_with_larger_raw_layer: List[Dataset]) -> None:
    """
    For datasets with a raw layer, and where raw.var is bigger than var,
    `open_anndata` should return a modified normalized layer
    (both var and X) that matches the size of raw and is "padded" accordingly.
    """
    result = list(open_anndata(".", datasets_with_larger_raw_layer))
    assert len(result) == 1
    _, h5ad = result[0]

    # Check that the var has a new row with feature_is_filtered=True and unknown
    # for the two other parameters
    assert h5ad.var.shape == (4, 4)
    added_var = h5ad.var.loc["homo_sapiens_d"]
    assert added_var.feature_is_filtered
    assert added_var.feature_name == "unknown"
    assert added_var.feature_reference == "unknown"

    # raw.var should not have feature_is_filtered at all
    assert h5ad.raw.shape == (4, 4)
    assert not h5ad.raw.var.feature_is_filtered.all()

    # The X matrix should have the same size as raw.X
    assert h5ad.X.shape == h5ad.raw.X.shape

    # The 4th column (correspnding to the added gene) is all zeros
    added_col = h5ad.X.todense()[:, 3]
    assert np.array_equal(added_col.A1, np.zeros(4))


def test_make_anndata_cell_filter(h5ad_simple: ad.AnnData) -> None:
    func = make_anndata_cell_filter({})  # type: ignore
    filtered_h5ad = func(h5ad_simple)
    assert h5ad_simple.var.equals(filtered_h5ad.var)
    assert h5ad_simple.obs.equals(filtered_h5ad.obs)
    assert np.array_equal(h5ad_simple.X.todense(), filtered_h5ad.X.todense())


def test_make_anndata_cell_filter_filters_out_organoids_cell_culture(
    h5ad_with_organoids_and_cell_culture: ad.AnnData,
) -> None:
    func = make_anndata_cell_filter({})  # type: ignore
    filtered_h5ad = func(h5ad_with_organoids_and_cell_culture)
    assert h5ad_with_organoids_and_cell_culture.var.equals(filtered_h5ad.var)
    assert filtered_h5ad.obs.shape[0] == 2


def test_make_anndata_cell_filter_organism(h5ad_with_organism: ad.AnnData) -> None:
    func = make_anndata_cell_filter({"organism_ontology_term_id": ORGANISMS[0].organism_ontology_term_id})  # type: ignore
    filtered_h5ad = func(h5ad_with_organism)
    assert h5ad_with_organism.var.equals(filtered_h5ad.var)
    assert filtered_h5ad.obs.shape[0] == 3


def test_make_anndata_cell_filter_feature_biotype_gene(h5ad_with_feature_biotype: ad.AnnData) -> None:
    func = make_anndata_cell_filter({})  # type: ignore
    filtered_h5ad = func(h5ad_with_feature_biotype)
    assert h5ad_with_feature_biotype.obs.equals(filtered_h5ad.obs)
    assert filtered_h5ad.var.shape[0] == 3


def test_make_anndata_cell_filter_assay(h5ad_with_assays: ad.AnnData) -> None:
    func = make_anndata_cell_filter({"assay_ontology_term_ids": ["EFO:1234", "EFO:1235"]})  # type: ignore
    filtered_h5ad = func(h5ad_with_assays)
    assert filtered_h5ad.obs.shape[0] == 2
    assert list(filtered_h5ad.obs.index) == ["1", "3"]


def test_get_cellxgene_schema_version(h5ad_simple: ad.AnnData) -> None:
    version = get_cellxgene_schema_version(h5ad_simple)
    assert version == "3.0.0"
