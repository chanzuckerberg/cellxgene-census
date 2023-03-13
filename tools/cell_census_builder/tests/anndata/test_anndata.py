from typing import List

import anndata as ad
import numpy as np

from tools.cell_census_builder.anndata import open_anndata
from tools.cell_census_builder.datasets import Dataset


def test_open_anndata(datasets: List[Dataset]) -> None:
    """
    `open_anndata` should open the h5ads for each of the dataset in the argument,
    and yield both the dataset and the corresponding AnnData object.
    This test does not involve additional filtering steps.
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


def test_open_anndata_equalizes_raw_and_normalized(datasets_with_larger_raw_layer: List[Dataset]) -> None:
    """
    For datasets where the raw.var is bigger than var, `open_anndata` should return a modified normalized layer
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
