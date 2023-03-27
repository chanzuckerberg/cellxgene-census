from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cell_census_builder")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-unknown"
