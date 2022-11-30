import os.path
import urllib.parse
from typing import Optional

import s3fs
import tiledb
import tiledbsoma as soma

from .release_directory import CensusLocator, CensusReleaseDescription, get_release_description
from .util import uri_join

# TODO: temporary work-around for lack of contenxt/config in tiledbsoma.  Replace with soma
# `platform_config` when available.
DEFAULT_TILEDB_CONFIGURATION = {
    # https://docs.tiledb.com/main/how-to/configuration#configuration-parameters
    "py.init_buffer_bytes": 1 * 1024**3,
    "soma.init_buffer_bytes": 1 * 1024**3,
    "py.deduplicate": "true",
}


def _open_soma(description: CensusReleaseDescription) -> soma.Collection:
    locator = description["soma"]
    tiledb_config = {**DEFAULT_TILEDB_CONFIGURATION}
    s3_region = locator.get("s3_region", None)
    if s3_region is not None:
        tiledb_config["vfs.s3.region"] = locator["s3_region"]
    return soma.Collection(uri=locator["uri"], ctx=tiledb.Ctx(tiledb_config))


def open_soma(*, census_version: Optional[str] = "latest", uri: Optional[str] = None) -> soma.Collection:
    """
    Open the Cell Census by version (name) or URI, returning a soma.Collection containing
    the top-level census.

    TODO: add platform_config hook when it is further defined, allowing config overrides.
    """

    if uri is not None:
        return soma.Collection(uri=uri, ctx=tiledb.Ctx(DEFAULT_TILEDB_CONFIGURATION))

    if census_version is None:
        raise ValueError("Must specify either a cell census version or an explicit URI.")

    description = get_release_description(census_version)  # raises
    return _open_soma(description)


def get_source_h5ad_uri(dataset_id: str, *, census_version: str = "latest") -> CensusLocator:
    """
    Open the named version of the census, and return the URI for the dataset_id.

    This does not guarantee that the H5AD exists or is accessible to the user.

    Raises if dataset_id or census_version are unknown.
    """
    description = get_release_description(census_version)  # raises
    census = _open_soma(description)
    dataset = census["census_info"]["datasets"].read_as_pandas_all(value_filter=f"dataset_id == '{dataset_id}'")
    if len(dataset) == 0:
        raise KeyError("Unknown dataset_id")

    locator = description["h5ads"].copy()
    h5ads_base_uri = locator["uri"]
    dataset_h5ad_path = dataset.dataset_h5ad_path.iloc[0]
    locator["uri"] = uri_join(h5ads_base_uri, dataset_h5ad_path)
    return locator


def download_source_h5ad(dataset_id: str, to_path: str, *, census_version: str = "latest") -> None:
    """
    Download the source H5AD dataset, for the given dataset_id, to the user-specified
    file name.

    Will raise an error if the path already exists (i.e., will not overwrite
    an existing file), or is not a file.

    Parameters
    ----------
    dataset_id : str
        Fetch the source (original) H5AD associated with this dataset_id.
    to_path : str
        The file name where the downloaded H5AD will be written.  Must not already exist.
    census_version : str
        The census version tag. Defaults to ``latest``.

    Returns
    -------
    None

    See Also
    --------
    get_source_h5ad_uri : Look up the location of the source H5AD.

    Examples
    --------
    >>> download_source_h5ad("8e47ed12-c658-4252-b126-381df8d52a3d", to_path="/tmp/data.h5ad")

    """
    if os.path.exists(to_path):
        raise ValueError("Path exists - will not overwrite existing file.")
    if to_path.endswith("/"):
        raise ValueError("Specify to_path as a file name, not a directory name.")

    locator = get_source_h5ad_uri(dataset_id, census_version=census_version)
    protocol = urllib.parse.urlparse(locator["uri"]).scheme
    assert protocol == "s3"

    fs = s3fs.S3FileSystem(
        anon=True,
        cache_regions=True,
    )
    fs.get_file(locator["uri"], to_path)
