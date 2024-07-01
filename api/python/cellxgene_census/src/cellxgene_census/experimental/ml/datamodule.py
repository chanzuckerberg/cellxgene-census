# type: ignore
import functools
from typing import List

import numpy.typing as npt
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule

from .pytorch import Encoder, ExperimentDataPipe, experiment_dataloader


class BatchEncoder(Encoder):
    """An encoder that concatenates and encodes several obs columns."""

    def __init__(self, cols: list[str], name: str = "batch"):
        self.cols = cols
        from sklearn.preprocessing import LabelEncoder

        self._name = name
        self._encoder = LabelEncoder()

    def _join_cols(self, df: pd.DataFrame) -> pd.Series[str]:
        return functools.reduce(lambda a, b: a + b, [df[c].astype(str) for c in self.cols])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the obs DataFrame into a DataFrame of encoded values."""
        arr = self._join_cols(df)
        return self._encoder.transform(arr)  # type: ignore

    def inverse_transform(self, encoded_values: npt.ArrayLike) -> npt.ArrayLike:
        """Inverse transform the encoded values back to the original values."""
        return self._encoder.inverse_transform(encoded_values)  # type: ignore

    def fit(self, obs: pd.DataFrame) -> None:
        """Fit the encoder with obs."""
        arr = self._join_cols(obs)
        self._encoder.fit(arr.unique())

    @property
    def columns(self) -> List[str]:
        """Columns in `obs` that the encoder will be applied to."""
        return self.cols

    @property
    def name(self) -> str:
        """Name of the encoder."""
        return self._name

    @property
    def classes_(self) -> List[str]:
        """Classes of the encoder."""
        return self._encoder.classes_


class CensusSCVIDataModule(LightningDataModule):
    """Lightning data module for training an scVI model using the ExperimentDataPipe.

    Parameters
    ----------
    *args
        Positional arguments passed to
        :class:`~cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe`.
    batch_keys
        List of obs column names concatenated to form the batch column.
    train_size
        Fraction of data to use for training.
    split_seed
        Seed for data split.
    dataloader_kwargs
        Keyword arguments passed into
        :func:`~cellxgene_census.experimental.ml.pytorch.experiment_dataloader`.
    **kwargs
        Additional keyword arguments passed into
        :class:`~cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe`. Must not include
        ``obs_column_names``.
    """

    _TRAIN_KEY = "train"
    _VALIDATION_KEY = "validation"

    def __init__(
        self,
        *args,
        batch_keys: list[str] | None = None,
        train_size: float | None = None,
        split_seed: int | None = None,
        dataloader_kwargs: dict[str, any] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.datapipe_args = args
        self.datapipe_kwargs = kwargs
        self.batch_keys = batch_keys
        self.train_size = train_size
        self.split_seed = split_seed
        self.dataloader_kwargs = dataloader_kwargs or {}

    @property
    def batch_keys(self) -> list[str]:
        """List of obs column names concatenated to form the batch column."""
        if not hasattr(self, "_batch_keys"):
            raise AttributeError("`batch_keys` not set.")
        return self._batch_keys

    @batch_keys.setter
    def batch_keys(self, value: list[str] | None):
        if value is None or not isinstance(value, list):
            raise ValueError("`batch_keys` must be a list of strings.")
        self._batch_keys = value

    @property
    def obs_column_names(self) -> list[str]:
        """Passed to :class:`~cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe`."""
        if hasattr(self, "_obs_column_names"):
            return self._obs_column_names

        obs_column_names = []
        if self.batch_keys is not None:
            obs_column_names.extend(self.batch_keys)

        self._obs_column_names = obs_column_names
        return self._obs_column_names

    @property
    def split_seed(self) -> int:
        """Seed for data split."""
        if not hasattr(self, "_split_seed"):
            raise AttributeError("`split_seed` not set.")
        return self._split_seed

    @split_seed.setter
    def split_seed(self, value: int | None):
        if value is not None and not isinstance(value, int):
            raise ValueError("`split_seed` must be an integer.")
        self._split_seed = value or 0

    @property
    def train_size(self) -> float:
        """Fraction of data to use for training."""
        if not hasattr(self, "_train_size"):
            raise AttributeError("`train_size` not set.")
        return self._train_size

    @train_size.setter
    def train_size(self, value: float | None):
        if value is not None and not isinstance(value, float):
            raise ValueError("`train_size` must be a float.")
        elif value is not None and (value < 0.0 or value > 1.0):
            raise ValueError("`train_size` must be between 0.0 and 1.0.")
        self._train_size = value or 1.0

    @property
    def validation_size(self) -> float:
        """Fraction of data to use for validation."""
        if not hasattr(self, "_train_size"):
            raise AttributeError("`validation_size` not available.")
        return 1.0 - self.train_size

    @property
    def weights(self) -> dict[str, float]:
        """Passed to :meth:`~cellxgene_census.experimental.ml.ExperimentDataPipe.random_split`."""
        if not hasattr(self, "_weights"):
            self._weights = {self._TRAIN_KEY: self.train_size}
            if self.validation_size > 0.0:
                self._weights[self._VALIDATION_KEY] = self.validation_size
        return self._weights

    @property
    def datapipe(self) -> ExperimentDataPipe:
        """Experiment data pipe."""
        if not hasattr(self, "_datapipe"):
            encoder = BatchEncoder(self.obs_column_names)
            self._datapipe = ExperimentDataPipe(
                *self.datapipe_args,
                encoders=[encoder],
                **self.datapipe_kwargs,
            )
        return self._datapipe

    def setup(self, stage: str | None = None):
        """Set up the train and validation data pipes."""
        datapipes = self.datapipe.random_split(weights=self.weights, seed=self.split_seed)
        self._train_datapipe = datapipes[0]
        if self.validation_size > 0.0:
            self._validation_datapipe = datapipes[1]
        else:
            self._validation_datapipe = None

    def train_dataloader(self):
        """Training data loader."""
        return experiment_dataloader(self._train_datapipe, **self.dataloader_kwargs)

    def val_dataloader(self):
        """Validation data loader."""
        if self._validation_datapipe is not None:
            return experiment_dataloader(self._validation_datapipe, **self.dataloader_kwargs)

    @property
    def n_obs(self) -> int:
        """Number of observations in the query.

        Necessary in scvi-tools to compute a heuristic of ``max_epochs``.
        """
        return self.datapipe.shape[0]

    @property
    def n_vars(self) -> int:
        """Number of features in the query.
        Necessary in scvi-tools to initialize the actual layers in the model.
        """
        return self.datapipe.shape[1]

    @property
    def n_batch(self) -> int:
        """Number of unique batches (after concatenation of ``batch_keys``).
        Necessary in scvi-tools so that the model knows how to one-hot encode batches.
        """
        return self.get_n_classes("batch")

    def get_n_classes(self, key: str) -> int:
        """Return the number of classes for a given obs column."""
        return len(self.datapipe.obs_encoders[key].classes_)

    def on_before_batch_transfer(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        dataloader_idx: int,
    ) -> dict[str, torch.Tensor | None]:
        """Format the datapipe output with registry keys for scvi-tools."""
        X, obs = batch

        X_KEY: str = "X"
        BATCH_KEY: str = "batch"
        LABELS_KEY: str = "labels"

        return {
            X_KEY: X,
            BATCH_KEY: obs,
            LABELS_KEY: None,
        }
