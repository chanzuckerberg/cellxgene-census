from __future__ import annotations

from pathlib import Path
from typing import Optional

from tap import Tap

from .load import npy_ingest, soma_ingest, test_embedding


class CommonArgs(Tap):  # type: ignore[misc]
    cwd: Path = Path.cwd()  # Working directory
    verbose: int = 0  # Logging level
    metadata: str = "meta.yml"  # Metadata file name, as .json or .yaml
    skip_storage_version_check: bool = False  # Skip TileDB storage equivalence check
    census_uri: Optional[
        str
    ] = None  # override Census URI. If not specified, will look up using metadata `census_version` field.

    def configure(self) -> None:
        super().configure()
        self.add_argument("-v", "--verbose", action="count", default=1, help="Increase logging verbosity")


class IngestCommonArgs(CommonArgs):
    float_precision: int = 7  # mantissa bits to preserve (range 4 to 23)


class IngestSOMAEmbedding(IngestCommonArgs):
    """Ingest embedding from SOMA SparseNDArray."""

    soma_path: Path  # Embedding encoded as a SOMA SparseNDArray

    def configure(self) -> None:
        super().configure()
        self.set_defaults(ingestor=lambda config: soma_ingest(config.args.soma_path, config))


class IngestNPYEmbedding(IngestCommonArgs):
    """Ingest embedding from NPY files."""

    joinid_path: Path  # Embedding soma_joinids, either .txt or .npy file
    embedding_path: Path  # Embedding coordinates, .npy file

    def configure(self) -> None:
        super().configure()
        self.set_defaults(
            ingestor=lambda config: npy_ingest(config.args.joinid_path, config.args.embedding_path, config)
        )


class IngestTestEmbedding(IngestCommonArgs):
    """Generate a test embedding containing random values."""

    def configure(self) -> None:
        super().configure()
        self.set_defaults(
            ingestor=lambda config: test_embedding(config.metadata.n_embeddings, config.metadata.n_features, config)
        )


class InjectEmbedding(CommonArgs):
    """Add existing embedding to a Census build as an obsm layer."""

    census_write_path: Path  # Census writable (build) path
    obsm_key: str  # key to store embedding as in the obsm collection


class Arguments(Tap):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__(underscores_to_dashes=True)

    def configure(self) -> None:
        self.add_subparsers(dest="cmd", required=True)
        self.add_subparser("ingest-soma", IngestSOMAEmbedding, help="Ingest embedding from SOMA SparseNDArray")
        self.add_subparser("ingest-npy", IngestNPYEmbedding, help="Ingest embedding from NPY files")
        self.add_subparser("ingest-test", IngestTestEmbedding, help="Generate an embedding containing random values")
        self.add_subparser("validate", CommonArgs, help="Validate an existing embedding")
        self.add_subparser("qcplots", CommonArgs, help="Generate QC plots for an existing embedding")
        self.add_subparser("inject", InjectEmbedding, help="Add existing embedding to a Census build as an obsm layer")

    def path_fix(self, arg_name: str) -> None:
        if hasattr(self, arg_name):
            setattr(self, arg_name, self.cwd.joinpath(getattr(self, arg_name)))

    def process_args(self) -> None:
        """
        process_args only called for classes where parse_ars is called, i.e.
        not on sub-command classes. So do all sub-class process_arg work here.
        """

        # Validate cwd
        if not self.cwd.is_dir():
            raise ValueError("Must specify working directory")

        for arg_name in ["joinid_path", "embedding_path", "soma_path"]:
            self.path_fix(arg_name)
