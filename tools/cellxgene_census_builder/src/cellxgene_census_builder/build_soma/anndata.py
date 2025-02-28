import logging
from contextlib import AbstractContextManager
from functools import cached_property
from os import PathLike
from types import TracebackType
from typing import Any, NotRequired, Protocol, Self, TypedDict, cast

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sparse
from anndata.abc import CSCDataset, CSRDataset
from anndata.io import read_elem, sparse_dataset

from ..util import urlcat
from .datasets import Dataset
from .globals import CXG_SCHEMA_VERSION

logger = logging.getLogger(__name__)


class AnnDataFilterSpec(TypedDict):
    organism_ontology_term_id: str
    assay_ontology_term_ids: list[str]
    is_primary_data: NotRequired[bool]


# Indexing types
Index1D = slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer[Any]]  # slice, mask, or points
Index = Index1D | tuple[Index1D] | tuple[Index1D, Index1D]


def _slice_index(prev: Index1D, new: Index1D, length: int) -> slice | npt.NDArray[np.int64]:
    """Slice an index."""
    if isinstance(prev, slice):
        if isinstance(new, slice):
            # conveniently, ranges support indexing!
            rng = range(*prev.indices(length))[new]
            assert rng.stop >= 0
            return slice(rng.start, rng.stop, rng.step)
        else:
            idx = np.arange(*prev.indices(length))[new]
            return idx if len(idx) else slice(0, 0)
    elif isinstance(prev, np.ndarray):
        if prev.dtype == np.bool_:  # a mask
            prev = np.nonzero(prev)[0].astype(np.int64)
        idx = cast(npt.NDArray[np.int64], prev[new])
        return idx if len(idx) else slice(0, 0)

    # else confusion
    raise IndexError("Unsupported indexing types")


def _normed_index(idx: Index) -> tuple[Index1D, Index1D]:
    if not isinstance(idx, tuple):
        return idx, slice(None)
    elif len(idx) == 1:
        return idx[0], slice(None)
    elif len(idx) == 2:
        return idx
    else:
        raise IndexError("Indexing supported on two dimensions only")


class AnnDataProxy(AbstractContextManager["AnnDataProxy"]):
    """Recommend using `open_anndata()` rather than instantiating this class directly.

    AnnData-like proxy for the version 0.1.0 AnnData H5PY file encoding (aka H5AD).
    Used in lieu of the AnnData class to reduce memory overhead. Semantics very similar
    to anndata.read_h5ad(backed="r"), but with the following optimizations:

    * opening and indexing does not materialize X (including repeated indexing,
      which is an operation where AnnData always materializes X)
    * opening only materializes obs/var, never obsm/varm/obsp/varp (which in many
      datasets is very large and memory-intensive)
    * only one copy of the base obs/var data is maintained. All views are defined
      only by an index (either a Python range, or a numpy point index array)

    This effectively:
    * removes the overhead of reading unused slots such as obsm/varm/obsp/varp
    * removes the overhead when you need to index a view (where AnnData materializes)

    In the future, if we need additional slots accessible, we may need to add sub-proxies
    for uns, obs*, var*, etc. But at the moment, the Builder has no use for these so they
    are omitted.
    """

    _obs: pd.DataFrame
    _var: pd.DataFrame
    _X: h5py.Dataset | CSRDataset | CSCDataset
    _file: h5py.File | None

    def __init__(
        self,
        filename: str | PathLike[str],
        *,
        view_of: Self | None = None,
        obs_idx: slice | npt.NDArray[np.int64] | None = None,
        var_idx: slice | npt.NDArray[np.int64] | None = None,
        obs_column_names: tuple[str, ...] | None = None,
        var_column_names: tuple[str, ...] | None = None,
    ):
        self.filename = filename

        if view_of is None:
            self._file = h5py.File(self.filename, mode="r")
            self._obs, self._var, self._X = self._load_h5ad(obs_column_names, var_column_names)
            self._obs_idx: slice | npt.NDArray[np.int64] = slice(None)
            self._var_idx: slice | npt.NDArray[np.int64] = slice(None)
        else:
            self._file = None
            self._obs, self._var, self._X = (view_of._obs, view_of._var, view_of._X)
            assert obs_idx is not None
            assert var_idx is not None
            self._obs_idx = obs_idx
            self._var_idx = var_idx

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        if self._file:
            self._file.close()
        self._file = None

    @property
    def X(self) -> sparse.spmatrix | npt.NDArray[np.integer[Any] | np.floating[Any]]:
        # For CS*Dataset, slice first on the major axis, then on the minor, as
        # the underlying AnnData proxy is not performant when slicing on the minor
        # axis (or both simultaneously). Let SciPy handle the second axis.
        if isinstance(self._X, CSRDataset):
            X = self._X[self._obs_idx][:, self._var_idx]
        elif isinstance(self._X, CSCDataset):
            X = self._X[:, self._var_idx][self._obs_idx]
        else:
            X = self._X[self._obs_idx, self._var_idx]

        if isinstance(X, sparse.spmatrix) and not X.has_canonical_format:
            X.sum_duplicates()
        return X

    @cached_property
    def obs(self) -> pd.DataFrame:
        return self._obs.iloc[self._obs_idx]

    @cached_property
    def var(self) -> pd.DataFrame:
        return self._var.iloc[self._var_idx]

    @property
    def n_obs(self) -> int:
        if isinstance(self._obs_idx, slice):
            return len(range(*self._obs_idx.indices(len(self._obs.index))))
        return len(self.obs)

    @property
    def n_vars(self) -> int:
        if isinstance(self._var_idx, slice):
            return len(range(*self._var_idx.indices(len(self._var.index))))
        return len(self.var)

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_obs, self.n_vars

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Index) -> "AnnDataProxy":
        odx, vdx = _normed_index(key)
        odx = _slice_index(self._obs_idx, odx, self.n_obs)
        vdx = _slice_index(self._var_idx, vdx, self.n_vars)
        return AnnDataProxy(self.filename, view_of=self, obs_idx=odx, var_idx=vdx)

    def get_estimated_density(self) -> float:
        """Return an estimated density for the H5AD, based upon the full file density.
        This is NOT the density for any given slice.

        Approach: divide the whole file nnz by the product of the shape.

        Arbitarily picks density of 1.0 if the file is empty on either axis
        """
        # Use whole-file n_obs/n_vars, not the slice length
        n_obs = len(self._obs)
        n_vars = len(self._var)
        if n_obs * n_vars == 0:
            return 1.0

        nnz: int
        if isinstance(self._X, CSRDataset | CSCDataset):
            nnz = self._X.group["data"].size
        else:
            nnz = self._X.size

        return nnz / (n_obs * n_vars)

    def _load_dataframe(self, elem: h5py.Group, column_names: tuple[str, ...] | None) -> pd.DataFrame:
        # if reading all, just use the built-in
        if not column_names:
            return cast(pd.DataFrame, read_elem(elem))

        # else read each user-specified column/index separately, taking care to preserve the
        # original dataframe column ordering
        assert len(column_names) > 0
        assert (
            elem.attrs["encoding-type"] == "dataframe" and elem.attrs["encoding-version"] == "0.2.0"
        ), "Unsupported AnnData encoding-type or encoding-version - likely indicates file was created with an unsupported AnnData version"
        column_order = elem.attrs["column-order"]
        column_names_ordered = [c for c in column_order if c in column_names and c != "_index"]
        index: npt.NDArray[Any] | None = None
        if "_index" in column_names:
            index_col_name = elem.attrs["_index"]
            index = read_elem(elem[index_col_name])
        return pd.DataFrame({c: read_elem(elem[c]) for c in column_names_ordered}, index=index)

    def _load_h5ad(
        self, obs_column_names: tuple[str, ...] | None, var_column_names: tuple[str, ...] | None
    ) -> tuple[pd.DataFrame, pd.DataFrame, CSRDataset | CSCDataset | h5py.Dataset]:
        """A memory optimization to prevent reading unnecessary data from the H5AD.

        This includes skipping:
            * obsm/varm/obsp/varp
            * unused obs/var columns
            * reading both raw and !raw

        Could be done with AnnData, at the expense of time & space to to read unused slots.

        Semantics are equivalent of doing:

            adata = anndata.read_h5ad(filename, backed="r")
            var, X = (adata.raw.var, adata.raw.X) if adata.raw else (adata.var, adata.X)
            return adata.obs, var, X

        This code utilizes the AnnData on-disk spec and several experimental API (as of 0.10.0).
        Spec: https://anndata.readthedocs.io/en/latest/fileformat-prose.html
        """
        assert isinstance(self._file, h5py.File)

        # Known to be compatible with this AnnData file encoding
        assert (
            self._file.attrs["encoding-type"] == "anndata" and self._file.attrs["encoding-version"] == "0.1.0"
        ), "Unsupported AnnData encoding-type or encoding-version - likely indicates file was created with an unsupported AnnData version"

        # Verify we are reading the expected CxG schema version.
        if "schema_version" in self._file["uns"]:
            schema_version = read_elem(self._file["uns/schema_version"])
        else:
            schema_version = "UNKNOWN"
        if schema_version != CXG_SCHEMA_VERSION:
            raise ValueError(
                f"{self.filename} -- incorrect CxG schema version (got {schema_version}, expected {CXG_SCHEMA_VERSION})"
            )

        obs = self._load_dataframe(self._file["obs"], obs_column_names)
        if "raw" in self._file:
            var = self._load_dataframe(self._file["raw/var"], var_column_names)
            X = self._file["raw/X"]
        else:
            var = self._load_dataframe(self._file["var"], var_column_names)
            X = self._file["X"]

        if isinstance(X, h5py.Group):
            X = sparse_dataset(X)

        assert isinstance(obs, pd.DataFrame)
        assert isinstance(var, pd.DataFrame)
        return obs, var, X


# The minimum columns required to be able to filter an H5AD.  See `make_anndata_cell_filter` for details.
CXG_OBS_COLUMNS_MINIMUM_READ = ("assay_ontology_term_id", "organism_ontology_term_id", "tissue_type", "is_primary_data")
CXG_VAR_COLUMNS_MINIMUM_READ = ("feature_biotype", "feature_reference")


def open_anndata(
    dataset: str | Dataset,
    *,
    base_path: str | None = None,
    include_filter_columns: bool = False,
    obs_column_names: tuple[str, ...] | None = None,
    var_column_names: tuple[str, ...] | None = None,
    filter_spec: AnnDataFilterSpec | None = None,
) -> AnnDataProxy:
    """Open the dataset and return an AnnData-like AnnDataProxy object.

    Args:
        {obs,var}_column_names: if specified, determine which columns are loaded for the respective dataframes.
            If not specified, all columns of obs/var are loaded.
        include_filter_columns: if True, ensure that any obs/var columns required for H5AD filtering are included. If
            False (default), only load the columsn specified by the user.
    """
    h5ad_path = dataset.dataset_h5ad_path if isinstance(dataset, Dataset) else dataset
    h5ad_path = urlcat(base_path, h5ad_path) if base_path is not None else h5ad_path

    include_filter_columns = include_filter_columns or (filter_spec is not None)
    if include_filter_columns:
        obs_column_names = tuple(set(CXG_OBS_COLUMNS_MINIMUM_READ + (obs_column_names or ())))
        var_column_names = tuple(set(CXG_VAR_COLUMNS_MINIMUM_READ + (var_column_names or ())))

    adata = AnnDataProxy(h5ad_path, obs_column_names=obs_column_names, var_column_names=var_column_names)
    if filter_spec is not None:
        adata = make_anndata_cell_filter(filter_spec)(adata)

    return adata


class AnnDataFilterFunction(Protocol):
    def __call__(self, ad: AnnDataProxy) -> AnnDataProxy: ...


def make_anndata_cell_filter(filter_spec: AnnDataFilterSpec) -> AnnDataFilterFunction:
    """Return an anndata sliced/filtered for those cells/genes of interest.

    See: https://github.com/chanzuckerberg/cellxgene-census/blob/main/docs/cellxgene_census_schema.md

    obs filter:
    * not organoid or cell culture
    * Caller-specified assays only
    * Caller-specified taxa (obs.organism_ontology_term_id == '<user-supplied>')
    * Organism term ID value not equal to gene feature_reference value
    * Single organism

    var filter:
    * genes only  (var.feature_biotype == 'gene')
    * Single organism
    """
    organism_ontology_term_id = filter_spec.get("organism_ontology_term_id")
    assert isinstance(organism_ontology_term_id, str)
    assay_ontology_term_ids = filter_spec.get("assay_ontology_term_ids")
    assert isinstance(assay_ontology_term_ids, list)
    is_primary_data = filter_spec.get("is_primary_data", None)
    assert isinstance(is_primary_data, bool | None)

    def _filter(ad: AnnDataProxy) -> AnnDataProxy:
        """Filter observations and features per Census schema."""
        var_mask = ad.var.feature_biotype == "gene"
        obs_mask = ad.obs.tissue_type.isin(["tissue", "organoid"])

        # Handle multi-species edge case
        var_organisms = set(ad.var.feature_reference[var_mask].unique())
        obs_organisms = set(ad.obs.organism_ontology_term_id[obs_mask].unique())
        if len(var_organisms) > 1 and len(obs_organisms) > 1:
            # if multi-species on both axis -- drop everything
            logger.info(f"H5AD ignored - multi-species content on both axes: {ad.filename}")
            return ad[0:0]  # ie., drop all cells

        # Filter by the species specified in the filter-spec
        var_mask = var_mask & (ad.var.feature_reference == organism_ontology_term_id)
        obs_mask = obs_mask & (ad.obs.organism_ontology_term_id == organism_ontology_term_id)
        if assay_ontology_term_ids:
            obs_mask = obs_mask & ad.obs.assay_ontology_term_id.isin(assay_ontology_term_ids)

        # Filter by is_primary_data as specified in the filter-spec
        if is_primary_data is not None:
            obs_mask = obs_mask & (ad.obs.is_primary_data == is_primary_data)

        if not (var_mask.any() and obs_mask.any()):
            return ad[0:0]  # i.e., drop all cells

        return ad[
            slice(None) if obs_mask.all() else obs_mask.to_numpy(),
            slice(None) if var_mask.all() else var_mask.to_numpy(),
        ]

    return _filter
