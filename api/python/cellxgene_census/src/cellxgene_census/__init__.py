"""An API to facilitate use of the CZI Science CELLxGENE Census. The Census is a versioned container of single-cell data hosted at `CELLxGENE Discover`_.

The API is built on the `tiledbsoma` SOMA API, and provides a number of helper functions including:

    * Open a named version of the Census, for use with the SOMA API
    * Get a list of available Census versions, and for each version, a description
    * Get a slice of the Census as an AnnData, for use with ScanPy
    * Get the URI for, or directly download, underlying data in H5AD format

For more information on the API, visit the `cellxgene_census repo`_. For more information on SOMA, see the `tiledbsoma repo`_.

.. _CELLxGENE Discover:
    https://cellxgene.cziscience.com/

.. _cellxgene_census repo:
    https://github.com/chanzuckerberg/cellxgene-census/

.. _tiledbsoma repo:
    https://github.com/single-cell-data/TileDB-SOMA
"""

from importlib import metadata

from ._get_anndata import get_anndata
from ._open import (
    download_source_h5ad,
    get_default_soma_context,
    get_source_h5ad_uri,
    open_soma,
)
from ._presence_matrix import get_presence_matrix
from ._release_directory import (
    get_census_mirror_directory,
    get_census_version_description,
    get_census_version_directory,
)

try:
    __version__ = metadata.version("cellxgene_census")
except metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-unknown"

__all__ = [
    "download_source_h5ad",
    "get_anndata",
    "get_census_version_description",
    "get_census_version_directory",
    "get_census_mirror_directory",
    "get_default_soma_context",
    "get_presence_matrix",
    "get_source_h5ad_uri",
    "open_soma",
]
