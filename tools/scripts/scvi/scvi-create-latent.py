import gc

import anndata as ad
import cellxgene_census
import numpy as np
import pandas as pd
import tiledbsoma as soma
import yaml

import scvi

file = "scvi-config.yaml"

if __name__ == "__main__":
    with open(file) as f:
        config = yaml.safe_load(f)

    census = cellxgene_census.open_soma(census_version="latest")

    census_config = config.get("census")
    experiment_name = census_config.get("organism")
    obs_value_filter = census_config.get("obs_query")

    hv = pd.read_pickle("hv_genes.pkl")
    # fmt: off
    hv_idx = hv[hv is True].index
    # fmt: on

    query = census["census_data"][experiment_name].axis_query(
        measurement_name="RNA",
        var_query=soma.AxisQuery(coords=(list(hv_idx),)),
    )

    print("Converting to AnnData")
    ad = query.to_anndata(X_name="raw")

    ad.obs["batch"] = ad.obs["dataset_id"] + ad.obs["assay"] + ad.obs["suspension_type"] + ad.obs["donor_id"]

    del census, query, hv, hv_idx
    gc.collect()

    # TODO: ensure that the anndata we're loading doesn't have obs filtering

    model = scvi.model.SCVI.load("scvi.model.test", adata=ad)

    latent = model.get_latent_representation(ad)

    np.savetxt("latent.csv", latent, delimiter=",")
