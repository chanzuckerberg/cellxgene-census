import os
import time
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import requests
import urllib3
from scipy import sparse

T = TypeVar("T")


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


def shuffle(items: list[T], step: int) -> list[T]:
    """Shuffle (interleave) from each end of the list. Step param controls
    bias of selection from front and back, i.e., if if step==2, every other
    item will be selected from end of list, if step==3, every third item
    will come from the end of the list.

    Expected use: reorder a sorted list (e.g by size) so that it ends up as (for step==2):
    [largest, smallest, second-largest, second-smallest, third-largest, ...]

    It is possible that a random shuffle (e.g., reservoir) would be better for the
    build use case (see build_soma step 1), but that is TBD.
    """
    assert step > 0
    r = []
    for i in range(len(items)):
        if i % step == 0:
            r.append(items[-i // step - 1])
        else:
            r.append(items[(step - 1) * (i // step) + (i % step) - 1])
    return r
