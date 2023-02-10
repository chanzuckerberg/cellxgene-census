import os.path
import urllib.parse
from typing import Any, Dict, Optional

import s3fs
import tiledbsoma as soma

from .release_directory import CensusLocator, get_census_version_description
from .util import uri_join

DEFAULT_TILEDB_CONFIGURATION: Dict[str, Any] = {
    # https://docs.tiledb.com/main/how-to/configuration#configuration-parameters
    "py.init_buffer_bytes": 1 * 1024**3,
    "soma.init_buffer_bytes": 1 * 1024**3,
}


def _open_soma(locator: CensusLocator, context: Optional[soma.options.SOMATileDBContext] = None) -> soma.Collection:
    """Private. Merge config defaults and return open census as a soma Collection/context."""
    context = context or soma.options.SOMATileDBContext()
    s3_region = locator.get("s3_region")

    if not context:
        # if no user-defined context, cell_census defaults take precedence over SOMA defaults
        tiledb_config = {**DEFAULT_TILEDB_CONFIGURATION}
        if s3_region is not None:
            tiledb_config["vfs.s3.region"] = s3_region
        context = context.replace(tiledb_config=tiledb_config)
    else:
        # if specified, the user context takes precedence _except_ for AWS Region in locator
        if s3_region is not None:
            tiledb_config = context.tiledb_ctx.config()
            tiledb_config["vfs.s3.region"] = s3_region
            context = context.replace(tiledb_config=tiledb_config)

    return soma.open(locator["uri"], mode="r", soma_type=soma.Collection, context=context)


def open_soma(
    *,
    census_version: Optional[str] = "latest",
    uri: Optional[str] = None,
    context: Optional[soma.options.SOMATileDBContext] = None,
) -> soma.Collection:
    """
    Open the Cell Census by version or URI, returning a soma.Collection containing the
    top-level census.  Raises error if ``census_version`` is specified and unknown, or
    if neither ``uri`` or ``census_version`` are specified, or if the ``uri`` can not
    be opened [lifecycle: experimental].

    Parameters
    ----------
    census_version : Optional[str]
        The version of the Census, e.g., "latest"
    uri : Optional[str]
        The URI containing the Census SOMA objects. If specified, will take precedence
        over ``census_version`` parameter. An exception will be raised if the URI can
        not be opened.

    Returns
    -------
    soma.Collection : returns a SOMA Collection object. Can be used as a context manager, which
        will automatically close upon exit.

    Examples
    --------
    Open the default Cell Census version, using a context manager which will automatically
    close the census upon exit of the context.

    >>> with cell_census.open_soma() as census:
            ...

    Open and close:
    >>> census = cell_census.open_soma()
        ...
        census.close()

    Open a specific Cell Census by version:
    >>> with cell_census.open_soma("2022-12-31") as census:
            ...

    Open a Cell Census by S3 URI, rather than by version.
    >>> with cell_census.open_soma(uri="s3://bucket/path") as census:
            ...

    Open a Cell Census by path (file:// URI), rather than by version.
    >>> cell_census.open_soma(uri="/tmp/census") as census:
            ...
    """

    if uri is not None:
        return _open_soma({"uri": uri, "s3_region": None}, context)

    if census_version is None:
        raise ValueError("Must specify either a cell census version or an explicit URI.")

    description = get_census_version_description(census_version)  # raises
    return _open_soma(description["soma"], context)


def get_source_h5ad_uri(dataset_id: str, *, census_version: str = "latest") -> CensusLocator:
    """
    Open the named version of the census, and return the URI for the dataset_id. This
    does not guarantee that the H5AD exists or is accessible to the user. Raises an
    error if dataset_id or census_version are unknown [lifecycle: experimental].

    Parameters
    ----------
    dataset_id : str
        The dataset_id of interest
    census_version : Optional[str]
        The census version

    Returns
    -------
    CensusLocator : the URI and optional S3 region for the source H5AD

    Examples
    --------
    >>> cell_census.get_source_h5ad_uri("cb5efdb0-f91c-4cbd-9ad4-9d4fa41c572d")
    {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/h5ads/cb5efdb0-f91c-4cbd-9ad4-9d4fa41c572d.h5ad',
    's3_region': 'us-west-2'}
    """
    description = get_census_version_description(census_version)  # raises
    census = _open_soma(description["soma"])
    dataset = census["census_info"]["datasets"].read(value_filter=f"dataset_id == '{dataset_id}'").concat().to_pandas()
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
    file name. Will raise an error if the path already exists (i.e., will not overwrite
    an existing file), or is not a file [lifecycle: experimental].

    Parameters
    ----------
    dataset_id : str
        Fetch the source (original) H5AD associated with this dataset_id.
    to_path : str
        The file name where the downloaded H5AD will be written.  Must not already exist.
    census_version : str
        The census version name. Defaults to ``latest``.

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
