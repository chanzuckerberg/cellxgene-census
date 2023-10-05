import anndata as ad
import yaml

import scvi

file = "scvi-config.yaml"

if __name__ == "__main__":
    with open(file) as f:
        config = yaml.safe_load(f)

    print("Start SCVI run")

    # scvi settings
    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)

    ad.read_h5ad("anndata.h5ad")

    scvi.model.SCVI.setup_anndata(ad, batch_key="batch")

    model_config = config.get("model")
    n_hidden = model_config.get("n_hidden")
    n_latent = model_config.get("n_latent")
    n_layers = model_config.get("n_layers")
    dropout_rate = model_config.get("dropout_rate")

    print("Configure model")

    model = scvi.model.SCVI(ad, n_layers=n_layers, n_latent=n_latent, gene_likelihood="nb")

    train_config = config.get("train")
    max_epochs = train_config.get("max_epochs")
    batch_size = train_config.get("batch_size")
    train_size = train_config.get("train_size")
    early_stopping = train_config.get("early_stopping")
    devices = train_config.get("devices")

    trainer_config = train_config.get("trainer")

    training_plan_config = config.get("training_plan")

    scvi.settings.dl_num_workers = train_config.get("num_workers")

    print("Start training model")

    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_size=train_size,
        early_stopping=early_stopping,
        plan_kwargs=training_plan_config,
        strategy="ddp_find_unused_parameters_true",  # Required for Multi-GPU training.
        devices=devices,
        **trainer_config,
    )

    model.save("scvi.model")

    # torch.save(model.module.state_dict(), "scvi.model")
