import logging
from functools import cached_property
from os import PathLike
from typing import Any, Iterator, List, Optional, Protocol, Self, Tuple, TypedDict, Union, cast

import anndata
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sparse
from anndata.experimental import CSCDataset, CSRDataset, read_elem, sparse_dataset

from ..util import urlcat
from .datasets import Dataset
from .globals import CXG_SCHEMA_VERSION, FEATURE_REFERENCE_IGNORE

AnnDataFilterSpec = TypedDict(
    "AnnDataFilterSpec",
    {
        "organism_ontology_term_id": Optional[str],
        "assay_ontology_term_ids": Optional[List[str]],
    },
)


def open_anndata(
    base_path: str, datasets: Union[List[Dataset], Dataset], need_X: Optional[bool] = True, *args: Any, **kwargs: Any
) -> Iterator[Tuple[Dataset, anndata.AnnData]]:
    raise NotImplementedError()
    """
    Generator to open anndata in a given mode, and filter out those H5ADs which do not match our base
    criteria for inclusion in the census.

    Will localize non-local (eg s3) URIs to accomadate AnnData/H5PY requirement for a local file.

    Apply criteria to filter out H5ADs we don't want or can't process.  Also apply a set of normalization
    remainder of code expects, such as final/raw feature equivalence.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]

    for h5ad in datasets:
        path = urlcat(base_path, h5ad.dataset_h5ad_path)
        logging.debug(f"open_anndata: {path}")
        ad = anndata.read_h5ad(path, *args, **kwargs)

        # These are schema versions this code is known to work with. This is a
        # sanity check, which would be better implemented via a unit test at
        # some point in the future.
        assert CXG_SCHEMA_VERSION in ["4.0.0"]

        if h5ad.schema_version == "":
            h5ad.schema_version = get_cellxgene_schema_version(ad)
        if h5ad.schema_version != CXG_SCHEMA_VERSION:
            msg = f"H5AD {h5ad.dataset_h5ad_path} has unsupported schema version {h5ad.schema_version}, expected {CXG_SCHEMA_VERSION}"
            logging.error(msg)
            raise RuntimeError(msg)

        # Multi-organism datasets - any dataset with 2+ feature_reference organisms is ignored,
        # exclusive of values in FEATURE_REFERENCE_IGNORE. See also, cell filter for mismatched
        # cell/feature organism values.
        feature_reference_organisms = set(ad.var.feature_reference.unique()) - FEATURE_REFERENCE_IGNORE
        if len(feature_reference_organisms) > 1:
            logging.info(f"H5AD ignored due to multi-organism feature_reference: {h5ad.dataset_id}")
            continue

        # Schema 3.0 disallows cell filtering, but DOES allow feature/gene filtering.
        # The "census" specification requires that any filtered features be added back to
        # the final layer.
        #
        # NOTE: As currently defined, the Census only includes raw counts. Most H5ADs
        # contain multiple X layers, plus a number of other matrices (obsm, etc). These
        # other objects use substantial memory (and have other overhead when the AnnData
        # is sliced in the filtering step).
        #
        # To minimize that overhead, this code drops all AnnData fileds unused by the
        # Census, following the CXG 3 conventions: use raw if present, else X.
        #
        if ad.raw is not None:
            X = ad.raw.X
            missing_from_var = ad.raw.var.index.difference(ad.var.index)
            if len(missing_from_var) > 0:
                raw_var = ad.raw.var.loc[missing_from_var].copy()
                raw_var["feature_is_filtered"] = True
                # TODO - these should be looked up in the ontology
                raw_var["feature_name"] = "unknown"
                raw_var["feature_reference"] = "unknown"
                raw_var["feature_length"] = 0
                var = pd.concat([ad.var, raw_var])
            else:
                var = ad.raw.var

        else:
            X = ad.X
            var = ad.var

        if need_X and isinstance(X, (sparse.csr_matrix, sparse.csc_matrix)) and not X.has_canonical_format:
            logging.warning(f"H5AD with non-canonical X matrix at {path}")
            X.sum_duplicates()

        assert (
            not isinstance(X, (sparse.csr_matrix, sparse.csc_matrix)) or X.has_canonical_format
        ), f"Found H5AD with non-canonical X matrix in {path}"

        ad = anndata.AnnData(X=X if need_X else None, obs=ad.obs, var=var, raw=None, uns=ad.uns)
        assert not need_X or ad.X.shape == (len(ad.obs), len(ad.var))

        # TODO: In principle, we could look up missing feature_name, but for now, just assert they exist
        assert ((ad.var.feature_name != "") & (ad.var.feature_name != None)).all()  # noqa: E711

        yield (h5ad, ad)


class AnnDataFilterFunction(Protocol):
    def __call__(self, ad: anndata.AnnData, need_X: Optional[bool] = True) -> anndata.AnnData:
        ...


def make_anndata_cell_filter(filter_spec: AnnDataFilterSpec) -> AnnDataFilterFunction:
    raise NotImplementedError()
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

    def _filter(ad: anndata.AnnData, need_X: Optional[bool] = True) -> anndata.AnnData:
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
        assert len(feature_references) == 1  # else there is a bug in open_anndata
        feature_reference_organism_ontology_id = feature_references.pop()
        obs_mask = obs_mask & (ad.obs.organism_ontology_term_id == feature_reference_organism_ontology_id)

        # This does NOT slice raw on the var axis.
        # See https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.raw.html
        ad = ad[obs_mask, (ad.var.feature_biotype == "gene")]

        obs = ad.obs
        var = ad.var
        var.index.rename("feature_id", inplace=True)
        X = ad.X if need_X else None
        assert ad.raw is None

        # This discards all other ancillary state, eg, obsm/varm/....
        ad = anndata.AnnData(X=X, obs=obs, var=var)

        assert (
            X is None or isinstance(X, np.ndarray) or X.has_canonical_format
        ), "Found H5AD with non-canonical X matrix"

        assert X is None or np.all(X.sum(axis=1) > 0), "H5AD contains a cell with only zero valued counts"
        return ad

    return _filter


def get_cellxgene_schema_version(ad: anndata.AnnData) -> str:
    # cellxgene >=2.0
    if "schema_version" in ad.uns:
        # not sure why this is a nested array
        return str(ad.uns["schema_version"])

    # cellxgene 1.X
    if "version" in ad.uns:
        return str(ad.uns["version"]["corpora_schema_version"])

    return ""


"""

Exploratory reimplementation of open/filter AnnData with an eye toward memory
efficiency AND ability to operate in a fixed size memory budget for handling 
of X matrix.

"""
# Indexing types
Index1D = slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer[Any]]  # slice, mask, or points
Index = Index1D | tuple[Index1D] | tuple[Index1D, Index1D]


def _index_index(prev: Index1D, new: Index1D, length: int) -> slice | npt.NDArray[np.int64]:
    """Index an index"""
    if isinstance(prev, slice):
        if isinstance(new, slice):
            # conviently, ranges support indexing!
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
    AnnData-like proxy for the version 0.1.0 AnnData H5PY file encoding (aka H5AD).
    Used in lieu of the AnnData class to reduce memory overhead.
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
    ):
        self.filename = filename
        self.is_view = view_of is not None

        if view_of is None:
            self._obs, self._var, self._X = self._load_h5ad()
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
        # Do two separate slices, as the underlying AnnData CS*Dataset
        # does bad things when you do slicing on both axis simultaneously.
        # Rely on the fact that we need to be most selective on the obs
        # dimension, and let scipy.sparse handle the second slice.
        X = self._X[self._obs_idx][:, self._var_idx]
        if sparse.isspmatrix(X) and not X.has_canonical_format:
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

    def __getitem__(self, key: Index) -> "AnnDataProxy":
        odx, vdx = _normed_index(key)
        odx = _index_index(self._obs_idx, odx, self.n_obs)
        vdx = _index_index(self._var_idx, vdx, self.n_vars)
        return AnnDataProxy(self.filename, view_of=self, obs_idx=odx, var_idx=vdx)

    def _load_h5ad(self) -> tuple[pd.DataFrame, pd.DataFrame, CSRDataset | CSCDataset | h5py.Dataset]:
        # A memory optimization to prevent reading obsm/varm/obsp/varp which are often huge.
        # Could be done with AnnData, at the expense of time & space to read these other slots.

        file = h5py.File(self.filename, mode="r")

        # Known to be compatible with this AnnData file encodin
        assert file.attrs["encoding-type"] == "anndata" and file.attrs["encoding-version"] == "0.1.0"

        # Verify we are reading the expected CxG schema version.
        assert read_elem(file["uns/schema_version"]) == CXG_SCHEMA_VERSION

        obs = read_elem(file["obs"])
        if "raw" in file:
            var = read_elem(file["raw/var"])
            X = file["raw/X"]
            assert var.index.equals(read_elem(file["var"]).index)
        else:
            var = read_elem(file["var"])
            X = file["X"]

        if isinstance(X, h5py.Group):
            X = sparse_dataset(X)

        assert isinstance(obs, pd.DataFrame)
        assert isinstance(var, pd.DataFrame)
        # return anndata.AnnData(obs=obs, var=var, X=X)
        return obs, var, X


def open_anndata2(base_path: str, dataset: Dataset) -> AnnDataProxy:
    return AnnDataProxy(urlcat(base_path, dataset.dataset_h5ad_path))


class AnnDataFilterFunction2(Protocol):
    def __call__(self, ad: AnnDataProxy) -> AnnDataProxy:
        ...


def make_anndata_cell_filter2(filter_spec: AnnDataFilterSpec) -> AnnDataFilterFunction2:
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
            logging.info(f"H5AD ignored due to multi-organism feature_reference: {ad.filename}")
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
