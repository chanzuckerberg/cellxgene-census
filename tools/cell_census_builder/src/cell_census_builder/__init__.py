try:
    from importlib import metadata
except ImportError:
    # for python <=3.7
    import importlib_metadata as metadata  # type: ignore[no-redef]


try:
    __version__ = metadata.version("cell_census_builder")
except metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-unknown"
