import numpy as np
import pyarrow as pa
import tiledbsoma as soma

with open("latent-idx.npy", "rb") as f:
    idx = np.load(f)

with open("latent.npy", "rb") as f:
    latent = np.load(f)

print(idx.shape)
print(latent.shape)


with soma.Collection.create("./obsm") as obsm:
    # collection created. You can now add SOMA objects, e.g., a DenseNDArray.
    # New objects are returned open for write.
    X_scvi = obsm.add_new_sparse_ndarray("X_scvi", type=pa.float32(), shape=latent.shape)
    data = pa.SparseCOOTensor.from_dense_numpy(latent, dim_names=["soma_dim_0", "soma_dim_1"])
    X_scvi.write(data)

# example of opening collection to read an object back
with soma.open("./obsm") as obsm:
    data = obsm["X_scvi"].read()


# with soma.SparseNDArray.create(
#     "./test_sparse_ndarray", type=pa.float32(), shape=latent.shape
# ) as arr:
#     data = pa.SparseCOOTensor.from_dense_numpy(
#         latent, dim_names=["soma_dim_0", "soma_dim_1"]
#     )
#     arr.write(data)

# with soma.SparseNDArray.open("./test_sparse_ndarray") as arr:
#     print(arr.schema)
#     print('---')
#     print(arr.read().coos().concat())
