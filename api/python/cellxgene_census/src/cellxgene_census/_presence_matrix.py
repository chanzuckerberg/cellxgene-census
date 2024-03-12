# Copyright (c) 2022, Chan Zuckerberg Initiative
#
# Licensed under the MIT License.

"""Presence matrix methods.

Methods to retrieve the feature dataset presence matrix.
"""

import tiledbsoma as soma
from scipy import sparse

from ._experiment import _get_experiment


def get_presence_matrix(
    census: soma.Collection,
    organism: str,
    measurement_name: str = "RNA",
) -> sparse.csr_matrix:
    """Read the feature dataset presence matrix and return as a :class:`scipy.sparse.csr_array`. The
    returned sparse matrix is indexed on the first dimension by the dataset ``soma_joinid`` values,
    and on the second dimension by the ``var`` :class:`pandas.DataFrame` ``soma_joinid`` values.

    Args:
        census:
            The census from which to read the presence matrix.
        organism:
            The organism to query, usually one of ``"Homo sapiens"`` or ``"Mus musculus"``.
        measurement_name:
            The measurement object to query. Deafults to ``"RNA"``.

    Returns:
        A :class:`scipy.sparse.csr_array` object containing the presence matrix.

    Raises:
        ValueError: if the organism cannot be found.

    Lifecycle:
        maturing

    Examples:
        >>> get_presence_matrix(census, "Homo sapiens", "RNA")
        <321x60554 sparse array of type '<class 'numpy.uint8'>'
        with 6441269 stored elements in Compressed Sparse Row format>
    """
    exp = _get_experiment(census, organism)
    presence = exp.ms[measurement_name]["feature_dataset_presence_matrix"]
    return presence.read((slice(None),)).coos().concat().to_scipy().tocsr()
