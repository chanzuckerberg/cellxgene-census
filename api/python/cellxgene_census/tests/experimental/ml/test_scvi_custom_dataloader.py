from __future__ import annotations
from pprint import pprint
import numpy as np
import pandas as pd
import scvi
from scvi.data import synthetic_iid

#IMPORTANT COMMENT: IN ORDER FOR THIS TO RUN SCVI-TOOLS SHOULD BE BASED ON THE VERSION THAT EXISTS
#IN THIS BRANCH: https://github.com/scverse/scvi-tools/tree/ori-2907-custom-dataloader-registry
#PLEASE INSTALL IT FROM THERE.

def test_czi_custom_dataloader_scvi(save_path: str):
    # # should be ready for importing the cloned branch on a remote machine that runs github action
    # sys.path.insert(
    #     0,
    #     "/home/runner/work/scvi-tools/scvi-tools/cellxgene-census/api/python/cellxgene_census/src",
    # )
    # sys.path.insert(0, "src")
    import cellxgene_census
    import tiledbsoma as soma
    from cellxgene_census.experimental.ml import experiment_dataloader
    from cellxgene_census.experimental.ml.datamodule import CensusSCVIDataModule
    # from cellxgene_census.experimental.pp import highly_variable_genes

    # this test checks the local custom dataloder made by CZI and run several tests with it
    census = cellxgene_census.open_soma(census_version="stable")

    experiment_name = "mus_musculus"
    obs_value_filter = 'is_primary_data == True and tissue_general in ["kidney"] and nnz >= 3000'

    # This is under comments just to save time (selecting highly varkable genes):
    # top_n_hvg = 8000
    # hvg_batch = ["assay", "suspension_type"]
    #
    # # For HVG, we can use the `highly_variable_genes` function provided in the Census,
    # # which can compute HVGs in constant memory:
    #
    # query = census["census_data"][experiment_name].axis_query(
    #     measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
    # )
    # hvgs_df = highly_variable_genes(query, n_top_genes=top_n_hvg, batch_key=hvg_batch)
    #
    # hv = hvgs_df.highly_variable
    # hv_idx = hv[hv].index

    hv_idx = np.arange(100)  # just ot make it smaller and faster for debug

    # this is CZI part to be taken once all is ready
    batch_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]
    datamodule = CensusSCVIDataModule(
        census["census_data"][experiment_name],
        measurement_name="RNA",
        X_name="raw",
        obs_query=soma.AxisQuery(value_filter=obs_value_filter),
        var_query=soma.AxisQuery(coords=(list(hv_idx),)),
        batch_size=1024,
        shuffle=True,
        batch_keys=batch_keys,
        dataloader_kwargs={"num_workers": 0, "persistent_workers": False},
    )

    # table of genes should be filtered by soma_joinid - but we should keep the encoded indexes
    # This is nice to have and might be uses in the downstream analysis
    # var_df = census["census_data"][experiment_name].ms["RNA"].var.read().concat().to_pandas()
    # var_df = var_df.loc[var_df.soma_joinid.isin(
    #     list(datamodule.datapipe.var_query.coords[0] if datamodule.datapipe.var_query is not None
    #          else range(datamodule.n_vars)))]

    # basicaly we should mimiC everything below to any model census in scvi
    adata_orig = synthetic_iid()
    scvi.model.SCVI.setup_anndata(adata_orig, batch_key="batch")

    model = scvi.model.SCVI(adata_orig)
    model.train(max_epochs=1)

    dataloader = model._make_data_loader(adata_orig)
    _ = model.get_elbo(dataloader=dataloader)
    _ = model.get_marginal_ll(dataloader=dataloader)
    _ = model.get_reconstruction_error(dataloader=dataloader)
    _ = model.get_latent_representation(dataloader=dataloader)

    scvi.model.SCVI.prepare_query_anndata(adata_orig, reference_model=model)
    scvi.model.SCVI.load_query_data(adata_orig, reference_model=model)

    user_attributes = model._get_user_attributes()
    pprint(user_attributes)

    scvi.model._scvi.SCVI.setup_datamodule(datamodule)  # takes time
    model_census = scvi.model.SCVI(
        adata=None,
        registry=datamodule.registry,
        gene_likelihood="nb",
        encode_covariates=False,
    )

    pprint(datamodule.registry)

    max_epochs = 1

    model_census.train(
        datamodule=datamodule,
        max_epochs=max_epochs,
        early_stopping=False,
    )

    user_attributes_model_census = model_census._get_user_attributes()
    # # TODO: do we need to put inside
    # user_attributes_model_census = \
    #     {a[0]: a[1] for a in user_attributes_model_census if a[0][-1] == "_"}
    pprint(user_attributes_model_census)
    # dataloader_census = model_census._make_data_loader(datamodule.datapipe)
    # # this casus errors
    # _ = model_census.get_elbo(dataloader=dataloader_census)
    # _ = model_census.get_marginal_ll(dataloader=dataloader_census)
    # _ = model_census.get_reconstruction_error(dataloader=dataloader_census)
    # _ = model_census.get_latent_representation(dataloader=dataloader_census)

    model_census.save(save_path, overwrite=True)
    model_census2 = scvi.model.SCVI.load(save_path, adata=False)

    model_census2.train(
        datamodule=datamodule,
        max_epochs=max_epochs,
        early_stopping=False,
    )

    user_attributes_model_census2 = model_census2._get_user_attributes()
    pprint(user_attributes_model_census2)
    # dataloader_census2 = model_census2._make_data_loader()
    # this casus errors
    # _ = model_census2.get_elbo()
    # _ = model_census2.get_marginal_ll()
    # _ = model_census2.get_reconstruction_error()
    # _ = model_census2.get_latent_representation()

    # takes time
    adata = cellxgene_census.get_anndata(
        census,
        organism=experiment_name,
        obs_value_filter=obs_value_filter,
        var_coords=hv_idx,
    )

    # TODO: do we need to put inside (or is it alrady pre-made) - perhaps need to tell CZI
    adata.obs["batch"] = adata.obs[batch_keys].agg("".join, axis=1).astype("category")

    scvi.model.SCVI.prepare_query_anndata(adata, save_path)
    scvi.model.SCVI.load_query_data(registry=datamodule.registry, reference_model=save_path)

    scvi.model.SCVI.prepare_query_anndata(adata, model_census2)

    scvi.model.SCVI.setup_anndata(adata, batch_key="batch")  # needed?
    model_census3 = scvi.model.SCVI.load(save_path, adata=adata)

    model_census3.train(
        datamodule=datamodule,
        max_epochs=max_epochs,
        early_stopping=False,
    )

    user_attributes_model_census3 = model_census3._get_user_attributes()
    pprint(user_attributes_model_census3)
    _ = model_census3.get_elbo()
    _ = model_census3.get_marginal_ll()
    _ = model_census3.get_reconstruction_error()
    _ = model_census3.get_latent_representation()

    scvi.model.SCVI.prepare_query_anndata(adata, save_path, return_reference_var_names=True)
    scvi.model.SCVI.load_query_data(adata, save_path)

    scvi.model.SCVI.prepare_query_anndata(adata, model_census3)
    scvi.model.SCVI.load_query_data(adata, model_census3)

    datamodule_inference = CensusSCVIDataModule(
        census["census_data"][experiment_name],
        measurement_name="RNA",
        X_name="raw",
        obs_query=soma.AxisQuery(value_filter=obs_value_filter),
        var_query=soma.AxisQuery(coords=(list(hv_idx),)),
        batch_size=1024,
        shuffle=False,
        soma_chunk_size=50_000,
        batch_keys=["dataset_id", "assay", "suspension_type", "donor_id"],
        dataloader_kwargs={"num_workers": 0, "persistent_workers": False},
    )

    # Create a dataloder of a CZI module
    datapipe = datamodule_inference.datapipe
    dataloader = experiment_dataloader(datapipe, num_workers=0, persistent_workers=False)
    mapped_dataloader = (
        datamodule_inference.on_before_batch_transfer(tensor, None) for tensor in dataloader
    )
    # _ = model_census.get_elbo(dataloader=mapped_dataloader)
    # _ = model_census.get_marginal_ll(dataloader=mapped_dataloader)
    # _ = model_census.get_reconstruction_error(dataloader=mapped_dataloader)
    latent = model_census.get_latent_representation(dataloader=mapped_dataloader)

    emb_idx = datapipe._obs_joinids

    obs_soma_joinids = adata.obs["soma_joinid"]

    obs_indexer = pd.Index(emb_idx)
    idx = obs_indexer.get_indexer(obs_soma_joinids)
    # Reindexing is necessary to ensure that the cells in the embedding match the ones in
    # the anndata object.
    adata.obsm["scvi"] = latent[idx]

    # #We can now generate the neighbors and the UMAP (tutorials)
    # sc.pp.neighbors(adata, use_rep="scvi", key_added="scvi")
    # sc.tl.umap(adata, neighbors_key="scvi")
    # sc.pl.umap(adata, color="dataset_id", title="SCVI")
    #
    # sc.pl.umap(adata, color="tissue_general", title="SCVI")
    #
    # sc.pl.umap(adata, color="cell_type", title="SCVI")


def test_czi_custom_dataloader_scanvi(save_path: str):
    # # should be ready for importing the cloned branch on a remote machine that runs github action
    # sys.path.insert(
    #     0,
    #     "/home/runner/work/scvi-tools/scvi-tools/cellxgene-census/api/python/cellxgene_census/src",
    # )
    # sys.path.insert(0, "src")
    import cellxgene_census
    import tiledbsoma as soma
    from cellxgene_census.experimental.ml.datamodule import CensusSCVIDataModule
    # from cellxgene_census.experimental.pp import highly_variable_genes

    # this test checks the local custom dataloder made by CZI and run several tests with it
    census = cellxgene_census.open_soma(census_version="stable")

    experiment_name = "mus_musculus"
    obs_value_filter = (
        'is_primary_data == True and tissue_general in ["kidney","liver"] and nnz >= 3000'
    )

    # This is under comments just to save time (selecting highly varkable genes):
    # top_n_hvg = 8000
    # hvg_batch = ["assay", "suspension_type"]
    #
    # # For HVG, we can use the `highly_variable_genes` function provided in the Census,
    # # which can compute HVGs in constant memory:
    #
    # query = census["census_data"][experiment_name].axis_query(
    #     measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
    # )
    # hvgs_df = highly_variable_genes(query, n_top_genes=top_n_hvg, batch_key=hvg_batch)
    #
    # hv = hvgs_df.highly_variable
    # hv_idx = hv[hv].index

    hv_idx = np.arange(100)  # just ot make it smaller and faster for debug

    # this is CZI part to be taken once all is ready
    batch_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]
    label_keys = ["tissue_general"]
    datamodule = CensusSCVIDataModule(
        census["census_data"][experiment_name],
        measurement_name="RNA",
        X_name="raw",
        obs_query=soma.AxisQuery(value_filter=obs_value_filter),
        var_query=soma.AxisQuery(coords=(list(hv_idx),)),
        batch_size=1024,
        shuffle=True,
        batch_keys=batch_keys,
        label_keys=label_keys,
        unlabeled_category="label_0",
        dataloader_kwargs={"num_workers": 0, "persistent_workers": False},
    )

    # table of genes should be filtered by soma_joinid - but we should keep the encoded indexes
    # This is nice to have and might be uses in the downstream analysis
    # var_df = census["census_data"][experiment_name].ms["RNA"].var.read().concat().to_pandas()
    # var_df = var_df.loc[var_df.soma_joinid.isin(
    #     list(datamodule.datapipe.var_query.coords[0] if datamodule.datapipe.var_query is not None
    #          else range(datamodule.n_vars)))]

    # scvi.model._scvi.SCVI.setup_datamodule(datamodule)  # takes time
    # model_census = scvi.model.SCVI(adata=None,
    #     registry=datamodule.registry,
    #     gene_likelihood="nb",
    #     encode_covariates=False,
    # )
    #
    # pprint(datamodule.registry)
    #
    max_epochs = 1
    #
    # model_census.train(
    #     datamodule=datamodule,
    #     max_epochs=max_epochs,
    #     early_stopping=False,
    # )

    scvi.model.SCANVI.setup_datamodule(datamodule)
    pprint(datamodule.registry)
    model = scvi.model.SCANVI(adata=None, registry=datamodule.registry, datamodule=datamodule)
    model.view_anndata_setup(datamodule)
    adata_manager = model.adata_manager
    pprint(adata_manager.registry)
    model.train(
        datamodule=datamodule, max_epochs=max_epochs, train_size=0.5, check_val_every_n_epoch=1
    )
    # logged_keys = model.history.keys()
    # assert len(model._labeled_indices) == sum(adata.obs["labels"] != "label_0")
    # assert len(model._unlabeled_indices) == sum(adata.obs["labels"] == "label_0")
    # assert "elbo_validation" in logged_keys
    # assert "reconstruction_loss_validation" in logged_keys
    # assert "kl_local_validation" in logged_keys
    # assert "elbo_train" in logged_keys
    # assert "reconstruction_loss_train" in logged_keys
    # assert "kl_local_train" in logged_keys
    # assert "validation_classification_loss" in logged_keys
    # assert "validation_accuracy" in logged_keys
    # assert "validation_f1_score" in logged_keys
    # assert "validation_calibration_error" in logged_keys
    # adata2 = synthetic_iid()
    # predictions = model.predict(adata2, indices=[1, 2, 3])
    # assert len(predictions) == 3
    # model.predict()
    # df = model.predict(adata2, soft=True)
    # assert isinstance(df, pd.DataFrame)
    # model.predict(adata2, soft=True, indices=[1, 2, 3])
    # model.get_normalized_expression(adata2)
    # model.differential_expression(groupby="labels", group1="label_1")
    # model.differential_expression(groupby="labels", group1="label_1", group2="label_2")

