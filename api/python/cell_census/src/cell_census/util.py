import re
import urllib.parse

import tiledbsoma as soma


def uri_join(base: str, url: str) -> str:
    """
    like urllib.parse.urljoin, but doesn't get confused by S3://
    """
    p_url = urllib.parse.urlparse(url)
    if p_url.netloc:
        return url

    p_base = urllib.parse.urlparse(base)
    path = urllib.parse.urljoin(p_base.path, p_url.path)
    parts = [p_base.scheme, p_base.netloc, path, p_url.params, p_url.query, p_url.fragment]
    return urllib.parse.urlunparse(parts)


def get_experiment(census: soma.Collection, organism: str) -> soma.Experiment:
    """
    Given a census soma.Collection, return the experiment for the named
    organism. Organism matching is somewhat flexible, attempting to map
    from human-friendly names to the underlying collection element name.

    Will raise a ValueError if unable to find the specified organism.

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
    >>> cell_census.get_experiment(census, 'homo sapiens')

    >>> cell_census.get_experiment(census, 'Homo sapiens')

    >>> cell_census.get_experiment(census, 'homo_sapiens')

    """
    # lower/snake case the organism name to find the experiment name
    exp_name = re.sub(r"[ ]+", "_", organism).lower()

    if exp_name not in census["census_data"]:
        raise ValueError(f"Unknown organism {organism} - does not exist")
    exp = census["census_data"][exp_name]
    if exp.soma_type != "SOMAExperiment":
        raise ValueError(f"Unknown organism {organism} - not a SOMA Experiment")

    return exp
