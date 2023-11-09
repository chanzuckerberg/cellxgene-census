from __future__ import annotations

from tap import Tap

from .load import csv_ingest, npy_ingest, soma_ingest, test_embedding


# Common across all sub-commands
class CommonArgs(Tap):  # type: ignore[misc]
    accession: str  # Accession ID
    metadata: str  # Metadata file name, as .json or .yaml
    verbose: int = 0  # Logging level

    def configure(self) -> None:
        self.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=1,
            help="Increase logging verbosity",
        )


class IngestArgs(CommonArgs):
    save_soma_to: str  # Output location


class SOMAEmbedding(IngestArgs):
    soma_uri: str  # Embedding encoded as a SOMA SparseNDArray

    def configure(self) -> None:
        super().configure()
        self.add_argument("soma_uri")
        self.set_defaults(ingestor=lambda args, metadata: soma_ingest(args.soma_uri, metadata))


class CSVEmbedding(IngestArgs):
    csv_uri: str  # Embedding encoded as a CSV (or TSV) file

    def configure(self) -> None:
        super().configure()
        self.add_argument("csv_uri")
        self.set_defaults(ingestor=lambda args, metadata: csv_ingest(args.csv_uri, metadata))


class NPYEmbedding(IngestArgs):
    joinid_uri: str  # Embedding soma_joinids
    embedding_uri: str  # Embedding coordinates

    def configure(self) -> None:
        super().configure()
        self.set_defaults(ingestor=lambda args, metadata: npy_ingest(args.joinid_uri, args.embedding_uri, metadata))


class TestEmbedding(IngestArgs):
    n_features: int = 2  # embedding dimensionality
    n_obs: int = 0  # number of cells to embed

    def configure(self) -> None:
        super().configure()
        self.set_defaults(ingestor=lambda args, metadata: test_embedding(args.n_obs, args.n_features, metadata))


class ValidateEmbedding(CommonArgs):
    uri: str  # Embedding URI

    def configure(self) -> None:
        super().configure()
        self.add_argument("uri")


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
