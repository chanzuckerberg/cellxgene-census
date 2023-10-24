
import anndata
import yaml
from sklearn.decomposition import IncrementalPCA

n_components = 50

file = "scvi-config.yaml"


def _gen_batches(n, batch_size, min_batch_size=0):
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)


if __name__ == "__main__":
    with open(file) as f:
        config = yaml.safe_load(f)

    adata_config = config["anndata"]
    batch_key = adata_config.get("batch_key")
    ad_filename = adata_config.get("model_filename")

    ad = anndata.read_h5ad(ad_filename)

    X = ad.X

    print("Shape: ", X.shape)

    print("Starting PCA")

    n_samples, n_features = X.shape
    batch_size = 1000  # 5 * n_features

    transformer = IncrementalPCA(n_components=n_components, batch_size=200)

    for i, batch in enumerate(_gen_batches(n_samples, batch_size, min_batch_size=n_components or 0)):
        print(batch)
        X_batch = X[batch]
        print(f"Computing batch {i}, shape {X_batch.shape}")
        X_batch = X_batch.toarray()
        transformer.partial_fit(X_batch)

    print(transformer)

    # transformer.fit(ad.X)
    # print(transformer)
    # print(transformer.components_)

    # with open('pca.npy', 'wb') as f:
    # np.save(f, transformer.components_)

    # pca = scanpy.tl.pca(ad, n_comps=n_components)

    # with open('pca.npy', 'wb') as f:
    #     np.save(f, ad.obsm["X_pca"])

    # print(ad.obsm["X_pca"].shape)


# Run PCA on the original matrix (the same data you input in SCVI. do not use raw counts, but use log transform data. see sc.pp.log1p() in scanpy). how many components? 50. plot the variance of each component

# Run SCVI on the same matrix (raw counts) and get the latent space. The number of latent dimensions does not need match the number of PCA components.

# Once you have both you can run scib metrics
