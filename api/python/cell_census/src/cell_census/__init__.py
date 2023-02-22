"""
Cell Census convience API

An API to facilitate use of the CZI Science Cell Census. The Cell Census is a versioned container of single-cell data hosted at [CELLxGENE Discover](https://cellxgene.cziscience.com/).

The API is built on the `tiledbsoma` SOMA API, and provides a number of helper functions including:

    * Open a named version of the Cell Census, for use with the SOMA API
    * Get a list of available Cell Census versions, and for each version, a description
    * Get a slice of the Cell Census as an AnnData, for use with ScanPy
    * Get the URI for, or directly download, underlying data in H5AD format

For more information on the API, visit the [cell_census repo](https://github.com/chanzuckerberg/cell-census/). For more information on SOMA, see the [tiledbsoma repo](https://github.com/single-cell-data/TileDB-SOMA).
"""

try:
    from importlib import metadata
except ImportError:
    # for python <=3.7
    import importlib_metadata as metadata  # type: ignore[no-redef]

from ._get_anndata import get_anndata
from ._open import download_source_h5ad, get_source_h5ad_uri, open_soma
from ._presence_matrix import get_presence_matrix
from ._release_directory import get_census_version_description, get_census_version_directory

try:
    __version__ = metadata.version("cell_census")
except metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-unknown"

__all__ = [
    "download_source_h5ad",
    "get_anndata",
    "get_census_version_description",
    "get_census_version_directory",
    "get_presence_matrix",
    "get_source_h5ad_uri",
    "open_soma",
]
