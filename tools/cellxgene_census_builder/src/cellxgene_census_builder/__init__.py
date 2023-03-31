from importlib import metadata

try:
    __version__ = metadata.version("cellxgene_census_builder")
except metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-unknown"
