# Copyright (c) 2022, Chan Zuckerberg Initiative
#
# Licensed under the MIT License.

"""Open census and related datasets.

Contains methods to open publicly hosted versions of Census object and access its source datasets.
"""
import logging
import os.path
import urllib.parse
from typing import Any, Dict, Optional, get_args

import s3fs
import tiledbsoma as soma

from ._release_directory import (
    CensusLocator,
    CensusMirror,
    Provider,
    ResolvedCensusLocator,
    _get_census_mirrors,
    get_census_version_description,
)
from ._util import _uri_join

DEFAULT_CENSUS_VERSION = "stable"

DEFAULT_TILEDB_CONFIGURATION: Dict[str, Any] = {
    # https://docs.tiledb.com/main/how-to/configuration#configuration-parameters
    "py.init_buffer_bytes": 1 * 1024**3,
    "soma.init_buffer_bytes": 1 * 1024**3,
    # S3 requests should not be signed, since we want to allow anonymous access
    "vfs.s3.no_sign_request": "true",
    "vfs.s3.region": "us-west-2",
}

api_logger = logging.getLogger("cellxgene_census")
api_logger.setLevel(logging.INFO)
api_logger.addHandler(logging.StreamHandler())


def _assert_mirror_supported(mirror: CensusMirror) -> None:
    """Verifies if the mirror is supported by this version of the census API.
    This method provides a proper error message in case an old version of the census
    tries to connect to an unsupported mirror.
    """
    if mirror["provider"] not in get_args(Provider):
        raise ValueError(
            f"Unsupported mirror provider: {mirror['provider']}. Try upgrading the cellxgene-census package to the latest version."
        )


def _resolve_census_locator(locator: CensusLocator, mirror: CensusMirror) -> ResolvedCensusLocator:
    _assert_mirror_supported(mirror)

    if locator.get("relative_uri"):
        uri = _uri_join(mirror["base_uri"], locator["relative_uri"])
        region = mirror["region"]
    else:
        uri = locator["uri"]
        region = locator.get("s3_region")
    return ResolvedCensusLocator(uri=uri, region=region, provider=mirror["provider"])


def _open_soma(
    locator: ResolvedCensusLocator,
    context: Optional[soma.options.SOMATileDBContext] = None,
) -> soma.Collection:
    """Private. Merge config defaults and return open census as a soma Collection/context."""
    # if no user-defined context, cellxgene_census defaults take precedence over SOMA defaults
    context = context or get_default_soma_context()

    # A locator's S3 bucket and region are intrinsically coupled, so ensure that the context always uses the
    # locator's region, even if the passed-in context specifies an alternate S3 region
    if locator["provider"] == "S3":
        context = context.replace(tiledb_config={"vfs.s3.region": locator.get("region")})

    return soma.open(locator["uri"], mode="r", soma_type=soma.Collection, context=context)


def get_default_soma_context(tiledb_config: Optional[Dict[str, Any]] = None) -> soma.options.SOMATileDBContext:
    """Return a :class:`tiledbsoma.SOMATileDBContext` with sensible defaults that can be further customized by the
    user. The customized context can then be passed to :func:`cellxgene_census.open_soma` with the ``context``
    argument or to :meth:`somacore.SOMAObject.open` with the ``context`` argument, such as
    :meth:`tiledbsoma.Experiment.open`. Use the :meth:`tiledbsoma.SOMATileDBContext.replace` method on the returned
    object to customize its settings further.

    Args:
        tiledb_config:
            A dictionary of TileDB configuration parameters. If specified, the parameters will override the
            defaults. If not specified, the default configuration will be returned.

    Returns:
        A :class:`tiledbsoma.SOMATileDBContext` object with sensible defaults.

    Examples:
        To reduce the amount of memory used by TileDB-SOMA I/O operations:

        .. highlight:: python
        .. code-block:: python

            ctx = cellxgene_census.get_default_soma_context(
                tiledb_config={"py.init_buffer_bytes": 128 * 1024**2,
                               "soma.init_buffer_bytes": 128 * 1024**2})
            c = census.open_soma(uri="s3://my-private-bucket/census/soma", context=ctx)

        To access a copy of the Census located in a private bucket that is located in a different S3 region, use:

        .. highlight:: python
        .. code-block:: python

            ctx = cellxgene_census.get_default_soma_context(
                tiledb_config={"vfs.s3.no_sign_request": "false",
                               "vfs.s3.region": "us-east-1"})
            c = census.open_soma(uri="s3://my-private-bucket/census/soma", context=ctx)

    Lifecycle:
        experimental
    """
    tiledb_config = dict(DEFAULT_TILEDB_CONFIGURATION, **(tiledb_config or {}))
    return soma.options.SOMATileDBContext().replace(tiledb_config=tiledb_config)


def open_soma(
    *,
    census_version: Optional[str] = DEFAULT_CENSUS_VERSION,
    mirror: Optional[str] = None,
    uri: Optional[str] = None,
    tiledb_config: Optional[Dict[str, Any]] = None,
    context: Optional[soma.options.SOMATileDBContext] = None,
) -> soma.Collection:
    """Open the Census by version or URI.

    Args:
        census_version:
            The version of the Census, e.g. ``"latest"`` or ``"stable"``. Defaults to ``"stable"``.
        mirror:
            The mirror used to retrieve the Census. If not specified, a suitable mirror
            will be chosen automatically.
        uri:
            The URI containing the Census SOMA objects. If specified, will take precedence
            over ``census_version`` parameter.
        tiledb_config:
            A dictionary of TileDB configuration parameters that will be used to open the SOMA object. Optional,
            defaults to ``None``. If specified, the parameters will override the default settings specified by
            ``get_default_soma_context().tiledb_config``. Only one of the ``tiledb_config`` and ``context`` params
            can be specified.
        context:
            A custom :class:`tiledbsoma.SOMATileDBContext` that will be used to open the SOMA object.
            Optional, defaults to ``None``. Only one of the ``tiledb_config`` and ``context`` params can be specified.

    Returns:
        A :class:`tiledbsoma.Collection` object containing the top-level census.
        It can be used as a context manager, which will automatically close upon exit.

    Raises:
        ValueError: if the census cannot be found, the URI cannot be opened, neither a URI
            or a version are specified, or an invalid mirror is provided.

    See Also:
        - :func:`get_source_h5ad_uri`: Look up the location of the source H5AD.
        - :func:`get_default_soma_context`: Get a default SOMA context that can be updated with custom settings and used
          as the ``context`` argument.

    Lifecycle:
        maturing

    Examples:
        Open the default Census version, using a context manager which will automatically
        close the Census upon exit of the context.

        >>> with cellxgene_census.open_soma() as census:
                ...

        Open and close:

        >>> census = cellxgene_census.open_soma()
            ...
            census.close()

        Open a specific Census by version:

        >>> with cellxgene_census.open_soma("2022-12-31") as census:
                ...

        Open a Census by S3 URI, rather than by version.

        >>> with cellxgene_census.open_soma(uri="s3://bucket/path") as census:
                ...

        Open a Census by path (file:// URI), rather than by version.

        >>> with cellxgene_census.open_soma(uri="/tmp/census") as census:
                ...

        Open a Census using a mirror.

        >>> with cellxgene_census.open_soma(mirror="s3-us-west-2") as census:
                ...

        Open a Census with a custom TileDB configuration setting.

        >>> with cellxgene_census.open_soma(tiledb_config={"py.init_buffer_bytes": 128 * 1024**2}) as census:
                ...
    """
    if tiledb_config is not None and context is not None:
        raise ValueError("Only one of tiledb_config and context can be specified.")

    if tiledb_config is not None:
        context = get_default_soma_context(tiledb_config=tiledb_config)

    if uri is not None:
        return _open_soma({"uri": uri, "region": None, "provider": "unknown"}, context)

    if census_version is None:
        raise ValueError("Must specify either a census version or an explicit URI.")

    mirrors = _get_census_mirrors()
    selected_mirror: CensusMirror
    if mirror is not None:
        if mirror not in mirrors:
            raise ValueError("Mirror not found.")
        selected_mirror = mirrors[mirror]  # type: ignore
    else:
        selected_mirror = mirrors[mirrors["default"]]  # type: ignore

    # TODO: Consider raising exceptions instead of issuing warnings, possibly introducing a "strict" mode to control the
    #  behavior
    try:
        description = get_census_version_description(census_version)  # raises
    except ValueError:
        raise ValueError(
            f'The "{census_version}" Census version is not valid. Use get_census_version_directory() to retrieve '
            f"available versions."
        ) from None

    if description.get("flags", {}).get("retracted", False):
        api_logger.warning(
            f"The \"{census_version}\" Census version has been retracted!\n{description['retraction']}."
            f'Use "stable" or "latest", or use get_census_version_directory() to retrieve valid versions.'
        )
    elif census_version == "stable":
        api_logger.info(
            f"The \"{census_version}\" release is currently {description['release_build']}. Specify "
            f"'census_version=\"{description['release_build']}\"' in future calls to open_soma() to ensure data "
            "consistency."
        )

    locator = _resolve_census_locator(description["soma"], selected_mirror)

    return _open_soma(locator, context)


def get_source_h5ad_uri(dataset_id: str, *, census_version: str = DEFAULT_CENSUS_VERSION) -> CensusLocator:
    """Open the named version of the census, and return the URI for the ``dataset_id``. This
    does not guarantee that the H5AD exists or is accessible to the user.

    Args:
        dataset_id:
            The ``dataset_id`` of interest.
        census_version:
            The census version. Defaults to ``"stable"``.

    Returns:
        A :class:`cellxgene_census._release_directory.CensusLocator` object that contains the URI and optional S3 region
        for the source H5AD.

    Raises:
        KeyError: if either ``dataset_id`` or ``census_version`` do not exist.

    Lifecycle:
        maturing

    Examples:
        >>> cellxgene_census.get_source_h5ad_uri("cb5efdb0-f91c-4cbd-9ad4-9d4fa41c572d")
        {'uri': 's3://cellxgene-data-public/cell-census/2022-12-01/h5ads/cb5efdb0-f91c-4cbd-9ad4-9d4fa41c572d.h5ad',
        's3_region': 'us-west-2'}
    """
    description = get_census_version_description(census_version)  # raises

    # For h5ads, it makes sense to use the default mirror, since the artifacts themselves won't be mirrored
    mirrors = _get_census_mirrors()
    selected_mirror: CensusMirror = mirrors[mirrors["default"]]  # type: ignore
    census_locator = _resolve_census_locator(description["soma"], selected_mirror)

    census = _open_soma(census_locator)
    dataset = census["census_info"]["datasets"].read(value_filter=f"dataset_id == '{dataset_id}'").concat().to_pandas()
    if len(dataset) == 0:
        raise KeyError("Unknown dataset_id")

    locator = description["h5ads"].copy()
    h5ads_base_uri = locator["uri"]
    dataset_h5ad_path = dataset.dataset_h5ad_path.iloc[0]
    locator["uri"] = _uri_join(h5ads_base_uri, dataset_h5ad_path)
    return locator


def download_source_h5ad(dataset_id: str, to_path: str, *, census_version: str = DEFAULT_CENSUS_VERSION) -> None:
    """Download the source H5AD dataset, for the given `dataset_id`, to the user-specified
    file name.

    Args:
        dataset_id
            Fetch the source (original) H5AD associated with this ``dataset_id``.
        to_path:
            The file name where the downloaded H5AD will be written. Must not already exist.
        census_version:
            The census version name. Defaults to ``"stable"``.

    Raises:
        ValueError: if the path already exists (i.e., will not overwrite an existing file), or is not a file.

    Lifecycle:
        maturing

    See Also:
        :func:`get_source_h5ad_uri`: Look up the location of the source H5AD.

    Examples:
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
