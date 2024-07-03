import pathlib
from typing import Any, cast

import anndata
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy import sparse

from cellxgene_census_builder.build_soma.anndata import (
    AnnDataFilterSpec,
    AnnDataProxy,
    make_anndata_cell_filter,
    open_anndata,
)
from cellxgene_census_builder.build_soma.datasets import Dataset
from cellxgene_census_builder.build_state import CensusBuildArgs

from ..conftest import GENE_IDS, ORGANISMS, get_anndata


def test_open_anndata(datasets: list[Dataset]) -> None:
    """`open_anndata` should open the h5ads for each of the dataset in the argument,
    and yield both the dataset and the corresponding AnnData object.
    This test does not involve additional filtering steps.
    The `datasets` used here have no raw layer.
    """

    def _todense(X: npt.NDArray[np.float32] | sparse.spmatrix) -> npt.NDArray[np.float32]:
        if isinstance(X, np.ndarray):
            return X
        else:
            return cast(npt.NDArray[np.float32], X.todense())

    result = [(d, open_anndata(d, base_path=".")) for d in datasets]
    assert len(result) == len(datasets) and len(datasets) > 0
    for i, (dataset, anndata_obj) in enumerate(result):
        assert dataset == datasets[i]
        opened_anndata = anndata.read_h5ad(dataset.dataset_h5ad_path)
        assert opened_anndata.obs.equals(anndata_obj.obs)
        assert opened_anndata.var.equals(anndata_obj.var)
        assert np.array_equal(_todense(opened_anndata.X), _todense(anndata_obj.X))

    # also check context manager
    with open_anndata(datasets[0], base_path=".") as ad:
        assert ad.n_obs == len(ad.obs)


def test_open_anndata_filters_out_datasets_with_mixed_feature_reference(
    datasets_with_mixed_feature_reference: list[Dataset],
) -> None:
    """Datasets with a "mixed" feature_reference will not be included by the filter pipeline"""
    ad_filter = make_anndata_cell_filter(
        {
            "organism_ontology_term_id": ORGANISMS[0].organism_ontology_term_id,
            "assay_ontology_term_ids": [],
        }
    )

    result = [ad_filter(open_anndata(d, base_path=".")) for d in datasets_with_mixed_feature_reference]
    assert all(len(ad) == 4 and ad.n_vars == len(GENE_IDS[0]) - 1 for ad in result)
    assert all((ad.var.feature_reference == ORGANISMS[0].organism_ontology_term_id).all() for ad in result)


def test_open_anndata_filters_out_wrong_schema_version_datasets(
    caplog: pytest.LogCaptureFixture,
    datasets_with_incorrect_schema_version: list[Dataset],
) -> None:
    """Datasets with a schema version different from `CXG_SCHEMA_VERSION` will not be included by `open_anndata`"""
    for dataset in datasets_with_incorrect_schema_version:
        with pytest.raises(ValueError, match="incorrect CxG schema version"):
            _ = open_anndata(dataset, base_path=".")


def test_make_anndata_cell_filter(tmp_path: pathlib.Path, h5ad_simple: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_simple)
    adata_simple = open_anndata(dataset, base_path=tmp_path.as_posix())

    func = make_anndata_cell_filter(
        {
            "organism_ontology_term_id": ORGANISMS[0].organism_ontology_term_id,
            "assay_ontology_term_ids": [],
        }
    )
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
    adata_with_organoids_and_cell_culture = open_anndata(dataset, base_path=tmp_path.as_posix())

    func = make_anndata_cell_filter(
        {
            "organism_ontology_term_id": ORGANISMS[0].organism_ontology_term_id,
            "assay_ontology_term_ids": [],
        }
    )
    filtered_adata_with_organoids_and_cell_culture = func(adata_with_organoids_and_cell_culture)

    assert adata_with_organoids_and_cell_culture.var.equals(filtered_adata_with_organoids_and_cell_culture.var)
    assert filtered_adata_with_organoids_and_cell_culture.obs.shape[0] == 2


def test_make_anndata_cell_filter_organism(tmp_path: pathlib.Path, h5ad_with_organism: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_with_organism)
    adata_with_organism = open_anndata(dataset, base_path=tmp_path.as_posix())

    func = make_anndata_cell_filter(
        {
            "organism_ontology_term_id": ORGANISMS[0].organism_ontology_term_id,
            "assay_ontology_term_ids": [],
        }
    )
    filtered_adata_with_organism = func(adata_with_organism)

    assert adata_with_organism.var.equals(filtered_adata_with_organism.var)
    assert filtered_adata_with_organism.obs.shape[0] == 3


def test_make_anndata_cell_filter_feature_biotype_gene(tmp_path: pathlib.Path, h5ad_with_feature_biotype: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_with_feature_biotype)
    adata_with_feature_biotype = open_anndata(dataset, base_path=tmp_path.as_posix())

    func = make_anndata_cell_filter(
        {
            "organism_ontology_term_id": ORGANISMS[0].organism_ontology_term_id,
            "assay_ontology_term_ids": [],
        }
    )
    filtered_adata_with_feature_biotype = func(adata_with_feature_biotype)

    assert adata_with_feature_biotype.obs.equals(filtered_adata_with_feature_biotype.obs)
    assert filtered_adata_with_feature_biotype.var.shape[0] == 3


def test_make_anndata_cell_filter_assay(tmp_path: pathlib.Path, h5ad_with_assays: str) -> None:
    dataset = Dataset(dataset_id="test", dataset_asset_h5ad_uri="test", dataset_h5ad_path=h5ad_with_assays)
    filter_spec = {
        "organism_ontology_term_id": ORGANISMS[0].organism_ontology_term_id,
        "assay_ontology_term_ids": ["EFO:1234", "EFO:1235"],
    }
    with open_anndata(dataset, base_path=tmp_path.as_posix(), filter_spec=filter_spec) as ad:
        assert ad.obs.shape[0] == 2
        assert list(ad.obs.index) == [0, 2]


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
        if isinstance(a, (sparse.csc_matrix, sparse.csr_matrix)):  # noqa: UP038
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
        #
        # empty slices
        [slice(0, 0, 1)],
        [np.array([], dtype=np.int64)],
        [slice(0, 0, 1), np.array([], dtype=np.int64)],
        [slice(0, 0, 1), np.array([], dtype=np.int64), slice(0, 0)],
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


@pytest.mark.parametrize("h5ad_simple", ("csr", "csc", "dense"), indirect=True)
def test_estimated_density(tmp_path: pathlib.Path, h5ad_simple: str) -> None:
    with open_anndata(h5ad_simple, base_path=tmp_path.as_posix()) as ad:
        density = ad.get_estimated_density()
        assert density == ad[1:].get_estimated_density()

    adata = anndata.read_h5ad(tmp_path / h5ad_simple)
    if isinstance(adata.X, sparse.spmatrix):
        assert density == adata.X.nnz / (adata.n_obs * adata.n_vars)
    else:
        assert density == 1.0


def test_empty_estimated_density(tmp_path: pathlib.Path) -> None:
    # make an empty AnnData
    path = tmp_path / "empty.h5ad"
    adata = anndata.AnnData(
        obs=pd.DataFrame(), var=pd.DataFrame({"feature_id": [0, 1, 2]}), X=sparse.csr_matrix((0, 3), dtype=np.float32)
    )
    adata.uns["schema_version"] = "5.1.0"
    adata.write_h5ad(path)

    with open_anndata(path) as ad:
        assert ad.get_estimated_density() == 1.0


def test_open_anndata_column_names(tmp_path: pathlib.Path, h5ad_simple: str) -> None:
    # check obs_column_names, var_column_names and include_filter_columns
    path = (tmp_path / h5ad_simple).as_posix()

    with open_anndata(path, obs_column_names=("cell_type_ontology_term_id", "sex", "_index")) as ad:
        assert set(ad.obs.keys()) == {"cell_type_ontology_term_id", "sex"}
        assert (ad.obs.index == ["1", "2", "3", "4"]).all()

    with open_anndata(path, var_column_names=("feature_name", "feature_reference", "_index")) as ad:
        assert set(ad.var.keys()) == {"feature_name", "feature_reference"}
        assert (ad.var.index == [f"homo_sapiens_{i}" for i in ["a", "b", "c", "d"]]).all()

    with open_anndata(path, include_filter_columns=True, obs_column_names=("cell_type",), var_column_names=()) as ad:
        assert set(ad.obs.keys()) == {"assay_ontology_term_id", "organism_ontology_term_id", "tissue_type", "cell_type"}
        assert set(ad.var.keys()) == {"feature_biotype", "feature_reference"}


def test_open_anndata_raw_X(tmp_path: pathlib.Path) -> None:
    # ensure we pick up raw if it is present
    path = tmp_path / "raw.h5ad"
    adata = anndata.AnnData(
        obs=pd.DataFrame({"cell_type": ["a", "b"]}, index=["A", "B"]),
        var=pd.DataFrame({"feature_id": [0, 1, 2]}),
        X=sparse.csr_matrix((2, 3), dtype=np.float32),
        raw={"X": sparse.csr_matrix((2, 4), dtype=np.float32)},
        uns={"schema_version": "5.1.0"},
    )
    adata.write_h5ad(path)

    with open_anndata(path) as ad:
        assert ad.X.shape == (2, 4)


HUMAN_FILTER_SPEC: AnnDataFilterSpec = {"organism_ontology_term_id": "NCBITaxon:9606", "assay_ontology_term_ids": []}
MOUSE_FILTER_SPEC: AnnDataFilterSpec = {"organism_ontology_term_id": "NCBITaxon:10090", "assay_ontology_term_ids": []}


@pytest.mark.parametrize(
    "organism_ontology_term_id,feature_reference,filter_spec,expected_shape",
    [
        (  # all human
            ["NCBITaxon:9606"] * 3,
            ["NCBITaxon:9606"] * 2,
            HUMAN_FILTER_SPEC,
            (3, 2),
        ),
        (  # all human
            ["NCBITaxon:9606"] * 3,
            ["NCBITaxon:9606"] * 2,
            MOUSE_FILTER_SPEC,
            (0, 2),
        ),
        (  # transgenic mouse
            ["NCBITaxon:10090"] * 5,
            ["NCBITaxon:9606", "NCBITaxon:10090", "NCBITaxon:10090"],
            HUMAN_FILTER_SPEC,
            (0, 3),
        ),
        (  # transgenic mouse
            ["NCBITaxon:10090"] * 5,
            ["NCBITaxon:9606", "NCBITaxon:10090", "NCBITaxon:10090"],
            MOUSE_FILTER_SPEC,
            (5, 2),
        ),
        (  # human with SARS
            ["NCBITaxon:9606"] * 7,
            ["NCBITaxon:9606"] * 2 + ["NCBITaxon:2697049"],
            HUMAN_FILTER_SPEC,
            (7, 2),
        ),
        (  # human with SARS
            ["NCBITaxon:9606"] * 7,
            ["NCBITaxon:9606"] * 2 + ["NCBITaxon:2697049"],
            MOUSE_FILTER_SPEC,
            (0, 3),
        ),
        (  # multi-species experiment,
            ["NCBITaxon:9606"] * 3 + ["NCBITaxon:10090"] * 5,
            ["NCBITaxon:9606"] * 2,
            HUMAN_FILTER_SPEC,
            (3, 2),
        ),
        (  # multi-species experiment,
            ["NCBITaxon:9606"] * 3 + ["NCBITaxon:10090"] * 5,
            ["NCBITaxon:9606"] * 2,
            MOUSE_FILTER_SPEC,
            (0, 2),
        ),
        (  # multi-species on both axes
            ["NCBITaxon:9606"] * 4 + ["NCBITaxon:10090"],
            ["NCBITaxon:9606"] * 2 + ["NCBITaxon:2697049"],
            HUMAN_FILTER_SPEC,
            (0, 3),
        ),
        (  # multi-species on both axes
            ["NCBITaxon:9606"] * 4 + ["NCBITaxon:10090"],
            ["NCBITaxon:9606"] * 2 + ["NCBITaxon:2697049"],
            MOUSE_FILTER_SPEC,
            (0, 3),
        ),
    ],
)
def test_multi_species_filter(
    tmp_path: pathlib.Path,
    organism_ontology_term_id: list[str],
    feature_reference: list[str],
    filter_spec: AnnDataFilterSpec,
    expected_shape: tuple[int, int],
) -> None:
    """Test all variations of multi-species filtering. Cell specifies defined by
    obs.organism_ontology_term_id, gene by var.feature_reference, both of which use
    UBERON ontology terms.

    Conditions:
    * the filter has a target species (e.g., NCBITaxon:9606, NCBITaxon:10090, etc)
    * the obs and var axis species may have no matches, partial matches or all matches with the filter target

    Expected results:
    * when both axes are multi-species, return empty result
    * when one or both axes are single-species, return slice that matches the filter

    IMPORTANT: if the filter results in no cells (obs) matching, the result is empty on the
    obs axes, and the var axis IS NOT FILTERED (as there is no point to doing so). So test
    the obs length before asserting anything about var.
    """

    n_obs = len(organism_ontology_term_id)
    n_vars = len(feature_reference)
    adata = anndata.AnnData(
        obs=pd.DataFrame(
            {"organism_ontology_term_id": organism_ontology_term_id, "tissue_type": "tissue"},
            index=[str(i) for i in range(n_obs)],
        ),
        var=pd.DataFrame(
            {"feature_reference": feature_reference, "feature_biotype": "gene"},
            index=[f"feature_{i}" for i in range(n_vars)],
        ),
        X=sparse.random(n_obs, n_vars, format="csr", dtype=np.float32),
        uns={"schema_version": "5.1.0"},
    )
    path = (tmp_path / "species.h5ad").as_posix()
    adata.write_h5ad(path)

    with open_anndata(path, filter_spec=filter_spec) as ad:
        assert ad.shape[0] == expected_shape[0] == ad.n_obs
        if ad.n_obs > 0:
            assert ad.shape == expected_shape == (ad.n_obs, ad.n_vars)
            assert (ad.obs.organism_ontology_term_id == filter_spec["organism_ontology_term_id"]).all()
            assert (ad.var.feature_reference == filter_spec["organism_ontology_term_id"]).all()
