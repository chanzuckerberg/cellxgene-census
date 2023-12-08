import anndata as ad
import scvi
import yaml
from lightning.pytorch.loggers import TensorBoardLogger

file = "scvi-config.yaml"

if __name__ == "__main__":
    with open(file) as f:
        config = yaml.safe_load(f)

    print("Start SCVI run")

    # scvi settings
    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)

    adata_config = config["anndata"]
    filename = adata_config.get("model_filename")

    adata = ad.read_h5ad(filename)

    scvi.model.SCVI.setup_anndata(adata, batch_key="batch")

    model_config = config.get("model")
    n_hidden = model_config.get("n_hidden")
    n_latent = model_config.get("n_latent")
    n_layers = model_config.get("n_layers")
    dropout_rate = model_config.get("dropout_rate")
    filename = model_config.get("filename")

    print("Configure model")

    model = scvi.model.SCVI(adata, n_layers=n_layers, n_latent=n_latent, gene_likelihood="nb", encode_covariates=True)

    train_config = config.get("train")
    max_epochs = train_config.get("max_epochs")
    batch_size = train_config.get("batch_size")
    train_size = train_config.get("train_size")
    early_stopping = train_config.get("early_stopping")
    devices = train_config.get("devices")
    multi_gpu = train_config.get("multi_gpu", False)

    trainer_config = train_config.get("trainer")

    training_plan_config = config.get("training_plan")

    if multi_gpu:
        scvi.settings.dl_num_workers = train_config.get("num_workers")
        strategy = "ddp_find_unused_parameters_true"
        devices = devices
    else:
        strategy = "auto"
        devices = 1

    print("Start training model")

    logger = TensorBoardLogger("tb_logs", name="my_model")

    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_size=train_size,
        early_stopping=early_stopping,
        plan_kwargs=training_plan_config,
        strategy=strategy,  # Required for Multi-GPU training.
        devices=devices,
        logger=logger,
        **trainer_config,
    )

    model.save(filename)
