import numpy as np
import tiledbsoma as soma
from scipy import sparse

from .util import get_experiment


def get_presence_matrix(
    census: soma.Collection,
    organism: str,
    measurement_name: str = "RNA",
) -> sparse.csr_array:
    """
    Read the gene presence matrix and return as a SciPy sparse CSR array (scipy.sparse.csr_array).

    The returned sparse matrix is indexed on the first dimension by the dataset ``soma_joinid`` values,
    and on the second dimension by the ``var`` DataFrame ``soma_joinid`` values.

    Parameters
    ----------
    census : soma.Collection
        The census from which to read the presence matrix.
    organism : str
        The organism to query, usually one of "Homo sapiens" or "Mus musculus"
    measurement_name : str, default 'RNA'
        The measurement object to query

    Returns
    -------
    scipy.sparse.csr_array - containing the presence matrix.

    Examples
    --------
    >>> get_presence_matrix(census, "Homo sapiens", "RNA")
    <321x60554 sparse array of type '<class 'numpy.uint8'>'
            with 6441269 stored elements in Compressed Sparse Row format>
    """

    exp = get_experiment(census, organism)
    presence = exp.ms[measurement_name].varp["dataset_presence_matrix"]

    # Read the entire presence matrix. It may be returned in incremental chunks if larger than
    # read buffers, so concatenate into a single scipy.sparse.sp_matrix.

    # TODO: TileDB-SOMA#596 when implemented, will simplify this

    arrow_sparse_tensors = [t for t in presence.read_sparse_tensor((slice(None),))]
    flat_arrays = [t.to_numpy() for t in arrow_sparse_tensors]
    data = np.concatenate(tuple(t[0] for t in flat_arrays))
    coords = np.concatenate(tuple(t[1] for t in flat_arrays))
    presence_matrix = sparse.coo_array(
        (data.flatten(), (coords.T[0].flatten(), coords.T[1].flatten())), shape=presence.shape
    ).tocsr()
    return presence_matrix
