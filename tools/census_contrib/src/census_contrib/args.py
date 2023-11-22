from __future__ import annotations

from pathlib import Path
from typing import Literal

from tap import Tap

from .load import csv_ingest, npy_ingest, soma_ingest, test_embedding


# Common across all sub-commands
class CommonArgs(Tap):  # type: ignore[misc]
    cwd: Path = Path.cwd()  # Working directory
    verbose: int = 0  # Logging level
    metadata: str = "meta.yml"  # Metadata file name, as .json or .yaml
    float_mode: Literal["scale", "trunc"] = "trunc"
    float_precision: int = 7  # mantissa bits to preserve (range 4 to 23)
    use_blockwise: bool = False  # Use the SOMA 1.5 blockwise iterators (requires tiledbsoma 1.5+)

    def configure(self) -> None:
        self.add_argument("-v", "--verbose", action="count", default=1, help="Increase logging verbosity")


class SOMAEmbedding(CommonArgs):
    soma_uri: Path  # Embedding encoded as a SOMA SparseNDArray

    def configure(self) -> None:
        super().configure()
        self.add_argument("soma_uri")
        self.set_defaults(ingestor=lambda args, metadata: soma_ingest(args.soma_uri, metadata))


class CSVEmbedding(CommonArgs):
    csv_uri: Path  # Embedding encoded as a CSV (or TSV) file

    def configure(self) -> None:
        super().configure()
        self.add_argument("csv_uri")
        self.set_defaults(ingestor=lambda args, metadata: csv_ingest(args.csv_uri, metadata))


class NPYEmbedding(CommonArgs):
    joinid_uri: Path  # Embedding soma_joinids, either .txt or .npy
    embedding_uri: Path  # Embedding coordinates

    def configure(self) -> None:
        super().configure()
        self.set_defaults(ingestor=lambda args, metadata: npy_ingest(args.joinid_uri, args.embedding_uri, metadata))


class TestEmbedding(CommonArgs):
    def configure(self) -> None:
        super().configure()
        self.set_defaults(
            ingestor=lambda args, metadata: test_embedding(metadata.n_embeddings, metadata.n_features, metadata)
        )


class ValidateEmbedding(CommonArgs):
    pass


class QCPlots(CommonArgs):
    pass


class Arguments(Tap):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__(underscores_to_dashes=True)

    def configure(self) -> None:
        self.add_subparsers(dest="cmd", help="Embedding source format", required=True)
        self.add_subparser("soma", SOMAEmbedding, help="Ingest embedding from SOMA SparseNDArray")
        self.add_subparser("csv", CSVEmbedding, help="Ingest embedding from CSV file")
        self.add_subparser("npy", NPYEmbedding, help="Ingest embedding from NPY files")
        self.add_subparser("test", TestEmbedding, help="Generate a random test embedding")
        self.add_subparser("validate", ValidateEmbedding, help="Validate existing embedding")
        self.add_subparser("qcplots", QCPlots, help="Generate QC plots")

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

        for arg_name in ["joinid_uri", "embedding_uri", "soma_uri", "csv_uri"]:
            self.path_fix(arg_name)
