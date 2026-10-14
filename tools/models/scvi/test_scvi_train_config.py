import contextlib
import copy
import os
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml


class TestScviTrainConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.script_dir = Path(__file__).resolve().parent
        with (self.script_dir / "scvi-config.yaml").open() as config_file:
            self.config = yaml.safe_load(config_file)

    def _run_training_script(self, config=None):
        adata = object()
        anndata = types.ModuleType("anndata")
        anndata.read_h5ad = Mock(return_value=adata)
        model = Mock(spec=["train", "save"])
        constructor = Mock(return_value=model)
        constructor.setup_anndata = Mock()
        scvi = types.ModuleType("scvi")
        scvi.__version__ = "test-stub"
        scvi.settings = types.SimpleNamespace()
        scvi.model = types.SimpleNamespace(SCVI=constructor)
        lightning = types.ModuleType("lightning")
        pytorch = types.ModuleType("lightning.pytorch")
        loggers = types.ModuleType("lightning.pytorch.loggers")
        loggers.TensorBoardLogger = Mock()
        lightning.pytorch = pytorch
        pytorch.loggers = loggers

        modules = {
            "anndata": anndata,
            "scvi": scvi,
            "lightning": lightning,
            "lightning.pytorch": pytorch,
            "lightning.pytorch.loggers": loggers,
        }
        config_patch = patch("yaml.safe_load", return_value=config) if config is not None else contextlib.nullcontext()
        previous_dir = Path.cwd()
        try:
            os.chdir(self.script_dir)
            with patch.dict(sys.modules, modules), config_patch:
                runpy.run_path(str(self.script_dir / "scvi-train.py"), run_name="__main__")
        finally:
            os.chdir(previous_dir)

        expected_config = self.config if config is None else config
        anndata.read_h5ad.assert_called_once_with(expected_config["anndata"]["model_filename"])
        constructor.setup_anndata.assert_called_once_with(adata, batch_key="batch")
        constructor.assert_called_once()
        self.assertEqual(constructor.call_args.args, (adata,))
        model.train.assert_called_once()
        model.save.assert_called_once_with(expected_config["model"]["filename"])
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["n_layers"], expected_config["model"]["n_layers"])
        self.assertEqual(kwargs["n_latent"], expected_config["model"]["n_latent"])
        self.assertEqual(kwargs["gene_likelihood"], "nb")
        self.assertIs(kwargs["encode_covariates"], False)
        return kwargs

    def test_n_hidden_is_forwarded_from_yaml(self) -> None:
        constructor_kwargs = self._run_training_script()
        self.assertIn(
            "n_hidden",
            constructor_kwargs,
            "SCVI constructor did not receive model.n_hidden from scvi-config.yaml",
        )
        self.assertEqual(constructor_kwargs["n_hidden"], self.config["model"]["n_hidden"])

    def test_missing_n_hidden_preserves_constructor_default(self) -> None:
        config = copy.deepcopy(self.config)
        del config["model"]["n_hidden"]
        self.assertNotIn("n_hidden", self._run_training_script(config))

    def test_none_n_hidden_preserves_constructor_default(self) -> None:
        config = copy.deepcopy(self.config)
        config["model"]["n_hidden"] = None
        self.assertNotIn("n_hidden", self._run_training_script(config))

    def test_alternate_n_hidden_is_forwarded(self) -> None:
        config = copy.deepcopy(self.config)
        config["model"]["n_hidden"] = self.config["model"]["n_hidden"] + 1
        constructor_kwargs = self._run_training_script(config)
        self.assertEqual(constructor_kwargs["n_hidden"], config["model"]["n_hidden"])

    def test_existing_model_dimensions_are_forwarded(self) -> None:
        constructor_kwargs = self._run_training_script()
        for name in ("n_layers", "n_latent"):
            with self.subTest(parameter=name):
                self.assertIn(name, constructor_kwargs)
                self.assertEqual(constructor_kwargs[name], self.config["model"][name])


if __name__ == "__main__":
    unittest.main()
