import pathlib
from typing import Any, List

import anndata as ad
import numpy as np
import numpy.typing as npt
import pytest
from cellxgene_census_builder.build_soma.anndata import AnnDataProxy, make_anndata_cell_filter, open_anndata
from cellxgene_census_builder.build_soma.datasets import Dataset
from cellxgene_census_builder.build_state import CensusBuildArgs
from scipy import sparse

from ..conftest import ORGANISMS, get_anndata


def test_open_anndata(datasets: List[Dataset]) -> None:
    """
    `open_anndata` should open the h5ads for each of the dataset in the argument,
    and yield both the dataset and the corresponding AnnData object.
    This test does not involve additional filtering steps.
    The `datasets` used here have no raw layer.
    """
    result = [(d, open_anndata(".", d)) for d in datasets]
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
    ad_filter = make_anndata_cell_filter({})
    result = [ad_filter(open_anndata(".", d)) for d in datasets_with_mixed_feature_reference]
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
            _ = open_anndata(".", dataset)


def test_make_anndata_cell_filter(tmp_path: pathlib.Path, h5ad_simple: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_simple)
    adata_simple = open_anndata(tmp_path.as_posix(), dataset)

    func = make_anndata_cell_filter({})
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
    adata_with_organoids_and_cell_culture = open_anndata(tmp_path.as_posix(), dataset)

    func = make_anndata_cell_filter({})
    filtered_adata_with_organoids_and_cell_culture = func(adata_with_organoids_and_cell_culture)

    assert adata_with_organoids_and_cell_culture.var.equals(filtered_adata_with_organoids_and_cell_culture.var)
    assert filtered_adata_with_organoids_and_cell_culture.obs.shape[0] == 2


def test_make_anndata_cell_filter_organism(tmp_path: pathlib.Path, h5ad_with_organism: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_with_organism)
    adata_with_organism = open_anndata(tmp_path.as_posix(), dataset)

    func = make_anndata_cell_filter({"organism_ontology_term_id": ORGANISMS[0].organism_ontology_term_id})
    filtered_adata_with_organism = func(adata_with_organism)

    assert adata_with_organism.var.equals(filtered_adata_with_organism.var)
    assert filtered_adata_with_organism.obs.shape[0] == 3


def test_make_anndata_cell_filter_feature_biotype_gene(tmp_path: pathlib.Path, h5ad_with_feature_biotype: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_with_feature_biotype)
    adata_with_feature_biotype = open_anndata(tmp_path.as_posix(), dataset)

    func = make_anndata_cell_filter({})
    filtered_adata_with_feature_biotype = func(adata_with_feature_biotype)

    assert adata_with_feature_biotype.obs.equals(filtered_adata_with_feature_biotype.obs)
    assert filtered_adata_with_feature_biotype.var.shape[0] == 3


def test_make_anndata_cell_filter_assay(tmp_path: pathlib.Path, h5ad_with_assays: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_with_assays)
    adata_with_assays = open_anndata(tmp_path.as_posix(), dataset, include_filter_columns=True)

    func = make_anndata_cell_filter({"assay_ontology_term_ids": ["EFO:1234", "EFO:1235"]})
    filtered_adata_with_assays = func(adata_with_assays)

    assert filtered_adata_with_assays.obs.shape[0] == 2
    assert list(filtered_adata_with_assays.obs.index) == [0, 2]


def make_h5ad_with_X_type(
    census_build_args: CensusBuildArgs,
    h5ad_path: str,
    X_conv: str,
    X_type: Any,
) -> npt.NDArray[Any] | sparse.spmatrix:
    census_build_args.h5ads_path.mkdir(parents=True, exist_ok=True)
    original_adata = get_anndata(ORGANISMS[0])
    assert isinstance(original_adata.X, sparse.csr_matrix)
    original_adata.X = getattr(original_adata.X, X_conv)()

    print(type(original_adata.X), X_type)
    assert isinstance(original_adata.X, X_type)
    original_adata.write_h5ad(h5ad_path)
    return original_adata.X


@pytest.mark.parametrize(
    "X_conv,X_type",
    [
        ("toarray", np.ndarray),
        ("tocsc", sparse.csc_matrix),
        ("tocsr", sparse.csr_matrix),
    ],
)
def test_AnnDataProxy_X_types(census_build_args: CensusBuildArgs, X_conv: str, X_type: Any) -> None:
    h5ad_path = f"{census_build_args.h5ads_path.as_posix()}/fmt_test_{X_conv}.h5ad"
    original_X = make_h5ad_with_X_type(census_build_args, h5ad_path, X_conv, X_type)

    adata = AnnDataProxy(h5ad_path)
    assert isinstance(adata.X, X_type)
    assert isinstance(adata[0:2].X, X_type)

    def _toarray(a: npt.NDArray[np.float32] | sparse.spmatrix) -> npt.NDArray[np.float32]:
        if isinstance(a, (sparse.csc_matrix, sparse.csr_matrix)):
            return a.toarray()  # type: ignore[no-any-return]
        else:
            return a

    assert np.array_equal(_toarray(adata.X), _toarray(original_X))
    assert np.array_equal(_toarray(adata[1:].X), _toarray(original_X[1:]))


@pytest.mark.parametrize(
    "slices",
    [
        # exercise different number of params
        [(slice(None), slice(None))],
        [slice(None)],
        [(slice(None),)],
        #
        # exercise masks
        [(np.zeros(4, dtype=np.bool_),)],
        [(np.ones(4, dtype=np.bool_),)],
        [(slice(1, 3), np.array([True, True, False, False]))],
        [(np.array([True, True, False, False]), slice(1, 3))],
        [np.array([True, False, True, False])],
        #
        # exercise slices
        [slice(2)],
        [slice(2, 4)],
        [slice(2, -2)],
        [(slice(None), slice(1, -1))],
        #
        # exercise combinations
        [(slice(1, -1), np.array([True, True, False, False]))],
        [(np.array([True, True, False, False]), slice(1, -1))],
        #
        # exercise multiple slices (slicing views)
        [slice(3), slice(2)],
        [np.array([True, True, True, False]), np.array([True, False, True])],
    ],
)
def test_AnnDataProxy_indexing(census_build_args: CensusBuildArgs, slices: Any) -> None:
    h5ad_path = f"{census_build_args.h5ads_path.as_posix()}/test.h5ad"
    original_X = make_h5ad_with_X_type(census_build_args, h5ad_path, "toarray", np.ndarray)

    adata = AnnDataProxy(h5ad_path)

    for slc in slices:
        assert np.array_equal(adata[slc].X, adata.X[slc])
        assert np.array_equal(adata[slc].X, original_X[slc])

        adata = adata[slc]
        original_X = original_X[slc]
