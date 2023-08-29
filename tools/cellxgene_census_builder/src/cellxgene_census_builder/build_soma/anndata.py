import logging
from typing import Any, Iterator, List, Optional, Protocol, Tuple, TypedDict, Union

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sparse

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
        assert CXG_SCHEMA_VERSION in ["3.1.0", "3.0.0"]

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

        ad = anndata.AnnData(X=X if need_X else None, obs=ad.obs, var=var, raw=None, uns=ad.uns, dtype=np.float32)
        assert not need_X or ad.X.shape == (len(ad.obs), len(ad.var))

        # TODO: In principle, we could look up missing feature_name, but for now, just assert they exist
        assert ((ad.var.feature_name != "") & (ad.var.feature_name != None)).all()  # noqa: E711

        yield (h5ad, ad)


class AnnDataFilterFunction(Protocol):
    def __call__(self, ad: anndata.AnnData, need_X: Optional[bool] = True) -> anndata.AnnData:
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
        ad = anndata.AnnData(X=X, obs=obs, var=var, dtype=np.float32)

        assert (
            X is None or isinstance(X, np.ndarray) or X.has_canonical_format
        ), "Found H5AD with non-canonical X matrix"

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
