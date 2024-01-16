import logging
from functools import cached_property
from os import PathLike
from typing import Any, List, Optional, Protocol, Self, Tuple, TypedDict, cast

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sparse
from anndata.experimental import CSCDataset, CSRDataset, read_elem, sparse_dataset

from ..util import urlcat
from .datasets import Dataset
from .globals import CXG_SCHEMA_VERSION, FEATURE_REFERENCE_IGNORE

logger = logging.getLogger(__name__)

AnnDataFilterSpec = TypedDict(
    "AnnDataFilterSpec",
    {
        "organism_ontology_term_id": Optional[str],
        "assay_ontology_term_ids": Optional[List[str]],
    },
)


# Indexing types
Index1D = slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer[Any]]  # slice, mask, or points
Index = Index1D | tuple[Index1D] | tuple[Index1D, Index1D]


def _slice_index(prev: Index1D, new: Index1D, length: int) -> slice | npt.NDArray[np.int64]:
    """Slice an index"""
    if isinstance(prev, slice):
        if isinstance(new, slice):
            # conveniently, ranges support indexing!
            rng = range(*prev.indices(length))[new]
            assert rng.stop >= 0
            return slice(rng.start, rng.stop, rng.step)
        else:
            return np.arange(*prev.indices(length))[new]
    elif isinstance(prev, np.ndarray):
        if prev.dtype == np.bool_:  # a mask
            prev = np.nonzero(prev)[0].astype(np.int64)
        return cast(npt.NDArray[np.int64], prev[new])

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


class AnnDataProxy:
    """
    Recommend using `open_anndata()` rather than instantiating this class directly.

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

    def __init__(
        self,
        filename: str | PathLike[str],
        *,
        view_of: Self | None = None,
        obs_idx: slice | npt.NDArray[np.int64] | None = None,
        var_idx: slice | npt.NDArray[np.int64] | None = None,
        obs_column_names: Optional[Tuple[str, ...]] = None,
        var_column_names: Optional[Tuple[str, ...]] = None,
    ):
        self.filename = filename

        if view_of is None:
            self._obs, self._var, self._X = self._load_h5ad(obs_column_names, var_column_names)
            self._obs_idx: slice | npt.NDArray[np.int64] = slice(None)
            self._var_idx: slice | npt.NDArray[np.int64] = slice(None)
        else:
            self._obs, self._var, self._X = (view_of._obs, view_of._var, view_of._X)
            assert obs_idx is not None
            assert var_idx is not None
            self._obs_idx = obs_idx
            self._var_idx = var_idx

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

    def _load_dataframe(self, elem: h5py.Group, column_names: Optional[Tuple[str, ...]]) -> pd.DataFrame:
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
        index: Optional[npt.NDArray[Any]] = None
        if "_index" in column_names:
            index_col_name = elem.attrs["_index"]
            index = read_elem(elem[index_col_name])
        return pd.DataFrame({c: read_elem(elem[c]) for c in column_names_ordered}, index=index)

    def _load_h5ad(
        self, obs_column_names: Optional[Tuple[str, ...]], var_column_names: Optional[Tuple[str, ...]]
    ) -> tuple[pd.DataFrame, pd.DataFrame, CSRDataset | CSCDataset | h5py.Dataset]:
        """
        A memory optimization to prevent reading unnecessary data from the H5AD. This includes
        skipping:
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

        file = h5py.File(self.filename, mode="r")

        # Known to be compatible with this AnnData file encoding
        assert (
            file.attrs["encoding-type"] == "anndata" and file.attrs["encoding-version"] == "0.1.0"
        ), "Unsupported AnnData encoding-type or encoding-version - likely indicates file was created with an unsupported AnnData version"

        # Verify we are reading the expected CxG schema version.
        schema_version = read_elem(file["uns/schema_version"])
        if schema_version != CXG_SCHEMA_VERSION:
            raise ValueError(
                f"{self.filename} -- incorrect CxG schema version (got {schema_version}, expected {CXG_SCHEMA_VERSION})"
            )

        obs = self._load_dataframe(file["obs"], obs_column_names)
        if "raw" in file:
            var = self._load_dataframe(file["raw/var"], var_column_names)
            X = file["raw/X"]
        else:
            var = self._load_dataframe(file["var"], var_column_names)
            X = file["X"]

        if isinstance(X, h5py.Group):
            X = sparse_dataset(X)

        assert isinstance(obs, pd.DataFrame)
        assert isinstance(var, pd.DataFrame)
        return obs, var, X


# The minimum columns required to be able to filter an H5AD.  See `make_anndata_cell_filter` for details.
CXG_OBS_COLUMNS_MINIMUM_READ = ("assay_ontology_term_id", "organism_ontology_term_id", "tissue_ontology_term_id")
CXG_VAR_COLUMNS_MINIMUM_READ = ("feature_biotype", "feature_reference")


def open_anndata(
    base_path: str,
    dataset: Dataset,
    *,
    include_filter_columns: bool = False,
    obs_column_names: Optional[Tuple[str, ...]] = None,
    var_column_names: Optional[Tuple[str, ...]] = None,
) -> AnnDataProxy:
    """
    Open the dataset and return an AnnData-like AnnDataProxy object.

    Args:
        {obs,var}_column_names: if specified, determine which columns are loaded for the respective dataframes.
            If not specified, all columns of obs/var are loaded.
        include_filter_columns: if True, ensure that any obs/var columns required for H5AD filtering are included. If
            False (default), only load the columsn specified by the user.
    """

    if include_filter_columns:
        obs_column_names = tuple(set(CXG_OBS_COLUMNS_MINIMUM_READ + (obs_column_names or ())))
        var_column_names = tuple(set(CXG_VAR_COLUMNS_MINIMUM_READ + (var_column_names or ())))

    return AnnDataProxy(
        urlcat(base_path, dataset.dataset_h5ad_path),
        obs_column_names=obs_column_names,
        var_column_names=var_column_names,
    )


class AnnDataFilterFunction(Protocol):
    def __call__(self, ad: AnnDataProxy) -> AnnDataProxy:
        ...


def make_anndata_cell_filter(filter_spec: AnnDataFilterSpec) -> AnnDataFilterFunction:
    """
    Return an anndata sliced/filtered for those cells/genes of interest.

    obs filter:
    * not organoid or cell culture
    * Caller-specified assays only
    * Caller-specified taxa (obs.organism_ontology_term_id == '<user-supplied>')
    * Organism term ID value not equal to gene feature_reference value

    var filter:
    * genes only  (var.feature_biotype == 'gene')
    """

    organism_ontology_term_id = filter_spec.get("organism_ontology_term_id", None)
    assay_ontology_term_ids = filter_spec.get("assay_ontology_term_ids", None)

    def _filter(ad: AnnDataProxy) -> AnnDataProxy:
        # Multi-organism datasets are dropped - any dataset with 2+ feature_reference organisms is ignored,
        # exclusive of values in FEATURE_REFERENCE_IGNORE. See also, cell filter for mismatched
        # cell/feature organism values.
        feature_reference_organisms = set(ad.var.feature_reference.unique()) - FEATURE_REFERENCE_IGNORE
        if len(feature_reference_organisms) > 1:
            logger.info(f"H5AD ignored due to multi-organism feature_reference: {ad.filename}")
            return ad[0:0]  # ie., drop all cells

        #
        # Filter cells per Census schema
        #
        obs_mask = ~(  # noqa: E712
            ad.obs.tissue_ontology_term_id.str.endswith(" (organoid)")
            | ad.obs.tissue_ontology_term_id.str.endswith(" (cell culture)")
        )

        if organism_ontology_term_id is not None:
            obs_mask = obs_mask & (ad.obs.organism_ontology_term_id == organism_ontology_term_id)
        if assay_ontology_term_ids is not None:
            obs_mask = obs_mask & ad.obs.assay_ontology_term_id.isin(assay_ontology_term_ids)

        # multi-organism dataset cell filter - exclude any cells where organism != feature_reference
        feature_references = set(ad.var.feature_reference.unique()) - FEATURE_REFERENCE_IGNORE
        assert len(feature_references) == 1  # else there is a bug in the test above
        feature_reference_organism_ontology_id = feature_references.pop()
        obs_mask = obs_mask & (ad.obs.organism_ontology_term_id == feature_reference_organism_ontology_id)

        #
        # Filter features per Census schema
        #
        var_mask = ad.var.feature_biotype == "gene"

        return ad[
            slice(None) if obs_mask.all() else obs_mask.to_numpy(),
            slice(None) if var_mask.all() else var_mask.to_numpy(),
        ]

    return _filter
