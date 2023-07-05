import logging
from typing import Any, Iterator, List, Optional, Protocol, Tuple, TypedDict, Union

import anndata
import numpy as np
import pandas as pd

from ..util import urlcat
from .datasets import Dataset
from .globals import CXG_SCHEMA_VERSION, CXG_SCHEMA_VERSION_IMPORT, FEATURE_REFERENCE_IGNORE

AnnDataFilterSpec = TypedDict(
    "AnnDataFilterSpec",
    {
        "organism_ontology_term_id": Optional[str],
        "assay_ontology_term_ids": Optional[List[str]],
    },
)


def open_anndata(
    base_path: str, datasets: Union[List[Dataset], Dataset], *args: Any, **kwargs: Any
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

        assert CXG_SCHEMA_VERSION == "3.0.0"
        if h5ad.schema_version == "":
            h5ad.schema_version = get_cellxgene_schema_version(ad)
        if h5ad.schema_version not in CXG_SCHEMA_VERSION_IMPORT:
            logging.error(f"H5AD has old schema version, skipping {h5ad.dataset_h5ad_path}")
            continue

        # Multi-organism datasets - any dataset with 2+ feature_reference organisms is ignored,
        # exclusive of values in FEATURE_REFERENCE_IGNORE. See also, cell filter for mismatched
        # cell/feature organism values.
        feature_reference_organisms = set(ad.var.feature_reference.unique()) - FEATURE_REFERENCE_IGNORE
        if len(feature_reference_organisms) > 1:
            logging.info(f"H5AD ignored due to multi-organism feature_reference: {h5ad.dataset_id}")
            continue

        # shape of raw and final must be same shape. Schema 2.0 disallows cell filtering,
        # but DOES allow feature/gene filtering. The "census" specification requires that
        # any filtered features be added back to the final layer.
        if ad.raw is not None:
            missing_from_var = ad.raw.var.index.difference(ad.var.index)
            if len(missing_from_var) > 0:
                raw_var = ad.raw.var.loc[missing_from_var].copy()
                raw_var["feature_is_filtered"] = True
                # TODO - these should be looked up in the ontology
                raw_var["feature_name"] = "unknown"
                raw_var["feature_reference"] = "unknown"
                new_var = pd.concat([ad.var, raw_var])
                if ad.isbacked:
                    ad = ad.to_memory()
                ad.X.resize(ad.n_obs, len(new_var))
                ad = anndata.AnnData(X=ad.X, obs=ad.obs, var=new_var, raw=ad.raw, dtype=np.float32)

        # Drop all of the fields we do not use (obsm, varm, uns, etc). Speeds up subsequent slicing
        # and reduces memory footprint.
        del ad.uns
        del ad.obsm
        del ad.obsp
        del ad.varm
        del ad.varp

        # sanity checks & expectations for any AnnData we can handle
        if ad.raw is not None:
            assert ad.X.shape == ad.raw.X.shape
            assert len(ad.raw.var) == len(ad.var)
            assert len(ad.raw.var.index.difference(ad.var.index)) == 0
            assert len(ad.var.index.difference(ad.raw.var.index)) == 0
        assert ad.X.shape == (len(ad.obs), len(ad.var))

        # TODO: In principle, we could look up missing feature_name, but for now, just assert they exist
        assert ((ad.var.feature_name != "") & (ad.var.feature_name != None)).all()  # noqa: E711

        yield (h5ad, ad)


class AnnDataFilterFunction(Protocol):
    def __call__(self, ad: anndata.AnnData, retain_X: Optional[bool] = True) -> anndata.AnnData:
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

    def _filter(ad: anndata.AnnData, retain_X: Optional[bool] = True) -> anndata.AnnData:
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
        X = ad.X if retain_X else None
        raw = ad.raw if retain_X and ad.n_obs > 0 else None

        if raw:
            # remove non-gene features
            mask = ad.raw.var.feature_biotype == "gene"
            raw = anndata.AnnData(X=ad.raw.X[:, mask], obs=ad.obs, var=ad.raw.var[mask], dtype=np.float32)

        # sanity checks
        if raw is not None:
            assert ad.var.index.difference(raw.var.index).empty
            assert raw.var.index.difference(ad.var.index).empty
            assert ad.X.shape == raw.X.shape

        # this dumps all other ancillary state, eg, obsm/varm/....
        ad = anndata.AnnData(X=X, obs=obs, var=var, raw=raw, dtype=np.float32)
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
