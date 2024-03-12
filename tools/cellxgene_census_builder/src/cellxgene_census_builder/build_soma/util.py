import os
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import requests
import urllib3
from scipy import sparse


def fetch_json(url: str, delay_secs: float = 0.0) -> object:
    s = requests.Session()
    retries = urllib3.util.Retry(
        backoff_factor=1,  # i.e., sleep for [0s, 2s, 4s, 8s, ...]
        status_forcelist=[500, 502, 503, 504],  # 5XX can occur on CDN failures, so retry
    )
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retries))
    response = s.get(url)
    response.raise_for_status()
    time.sleep(delay_secs)
    return response.json()


def is_nonnegative_integral(X: npt.NDArray[np.floating[Any]] | sparse.spmatrix) -> bool:
    """Return true if the matrix/array contains only positive integral values,
    False otherwise.
    """
    data = X if isinstance(X, np.ndarray) else X.data

    if np.signbit(data).any():
        return False
    elif np.any(~np.equal(np.mod(data, 1), 0)):
        return False
    else:
        return True


def get_git_commit_sha() -> str:
    """Returns the git commit SHA for the current repo."""
    # Try to get the git commit SHA from the COMMIT_SHA env variable
    commit_sha_var = os.getenv("COMMIT_SHA")
    if commit_sha_var is not None:
        return commit_sha_var

    import git  # Scoped import - this requires the git executable to exist on the machine

    # work around https://github.com/gitpython-developers/GitPython/issues/1349
    # by explicitly referencing git.repo.base.Repo instead of git.Repo
    repo = git.repo.base.Repo(search_parent_directories=True)
    hexsha: str = repo.head.object.hexsha
    return hexsha


def is_git_repo_dirty() -> bool:
    """Returns True if the git repo is dirty, i.e. there are uncommitted changes."""
    import git  # Scoped import - this requires the git executable to exist on the machine

    # work around https://github.com/gitpython-developers/GitPython/issues/1349
    # by explicitly referencing git.repo.base.Repo instead of git.Repo
    repo = git.repo.base.Repo(search_parent_directories=True)
    is_dirty: bool = repo.is_dirty()
    return is_dirty
