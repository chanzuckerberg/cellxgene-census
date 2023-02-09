import re

import tiledbsoma as soma


def get_experiment(census: soma.Collection, organism: str) -> soma.Experiment:
    """
    Given a census soma.Collection, return the experiment for the named organism.
    Organism matching is somewhat flexible, attempting to map from human-friendly
    names to the underlying collection element name.  Will raise a ValueError if
    unable to find the specified organism [lifecycle: experimental].

    Parameters
    ----------
    census - soma.Collection
        The census
    organism - str
        The organism name, eg., ``Homo sapiens``

    Returns
    -------
    soma.Experiment - the requested experiment.

    Examples
    --------
    >>> human = get_experiment(census, 'homo sapiens')

    >>> human = get_experiment(census, 'Homo sapiens')

    >>> human = get_experiment(census, 'homo_sapiens')
    """
    # lower/snake case the organism name to find the experiment name
    exp_name = re.sub(r"[ ]+", "_", organism).lower()

    if exp_name not in census["census_data"]:
        raise ValueError(f"Unknown organism {organism} - does not exist")
    exp = census["census_data"][exp_name]
    if exp.soma_type != "SOMAExperiment":
        raise ValueError(f"Unknown organism {organism} - not a SOMA Experiment")

    return exp
