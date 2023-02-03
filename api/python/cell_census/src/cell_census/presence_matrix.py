import tiledbsoma as soma
from scipy import sparse

from .experiment import get_experiment


def get_presence_matrix(
    census: soma.Collection,
    organism: str,
    measurement_name: str = "RNA",
) -> sparse.csr_matrix:
    """
    Read the gene presence matrix and return as a SciPy sparse CSR array
    (scipy.sparse.csr_array). The returned sparse matrix is indexed on the
    first dimension by the dataset ``soma_joinid`` values, and on the
    second dimension by the ``var`` DataFrame ``soma_joinid`` values
    [lifecycle: experimental].

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
    presence = exp.ms[measurement_name]["feature_dataset_presence_matrix"]
    return presence.read((slice(None),)).csrs().concat().to_scipy()
