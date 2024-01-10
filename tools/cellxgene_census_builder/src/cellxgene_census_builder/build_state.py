"""
Manage the configuration and dynamic build state for the Census build.

"""
import functools
import io
import os
import pathlib
from datetime import datetime
from typing import Any, Iterator, Mapping, Union

import psutil
import yaml
from attrs import define, field, fields, validators
from typing_extensions import Self

"""
Defaults for Census configuration.
"""

CENSUS_BUILD_CONFIG = "config.yaml"
CENSUS_BUILD_STATE = "state.yaml"


@define
class CensusBuildConfig:
    verbose: int = field(converter=int, default=1)
    log_dir: str = field(default="logs")
    log_file: str = field(default="build.log")
    reports_dir: str = field(default="reports")
    consolidate = field(converter=bool, default=True)
    disable_dirty_git_check = field(converter=bool, default=True)
    dryrun: bool = field(
        converter=bool, default=False
    )  # if True, will disable copy of data/logs/reports/release.json to S3 buckets. Will NOT disable local build, etc.
    #
    # Primary bucket location
    cellxgene_census_S3_path: str = field(default="s3://cellxgene-data-public/cell-census")
    #
    # Default mirror bucket location. Used for building legacy absolute URIs.
    cellxgene_census_default_mirror_S3_path: str = field(default="s3://cellxgene-census-public-us-west-2/cell-census")
    #
    # Replica bucket location
    cellxgene_census_S3_replica_path: str = field(default=None)
    logs_S3_path: str = field(default="s3://cellxgene-data-public-logs/builder")
    build_tag: str = field(default=datetime.now().astimezone().date().isoformat())
    #
    # Default multi-process. Memory scaling based on empirical tests.
    multi_process: bool = field(converter=bool, default=True)
    #
    # The memory budget used to determine appropriate parallelism in many steps of build.
    # Only set to a smaller number if you want to not use all available RAM.
    memory_budget: int = field(converter=int, default=psutil.virtual_memory().total)
    #
    # 'max_worker_processes' sets a limit on the number of worker processes to avoid running into
    # the max VM map kernel limit and other hard resource limits. The value specified here was
    # determined by empirical testing on 128 and 192 CPU boxes (e.g., r6a.48xlarge, r6i.32xlarge,
    # etc) using the default kernel config for Ubuntu 22.04
    max_worker_processes: int = field(converter=int, default=192)
    #
    # Host minimum resource validation
    host_validation_disable: int = field(
        converter=bool, default=False
    )  # if True, host validation checks will be skipped
    host_validation_min_physical_memory: int = field(converter=int, default=512 * 1024**3)  # 512GiB
    host_validation_min_swap_memory: int = field(converter=int, default=2 * 1024**4)  # 2TiB
    host_validation_min_free_disk_space: int = field(converter=int, default=int(1.8 * 1024**4))  # 1.8 TiB
    #
    # Release clean up
    release_cleanup_days: int = field(converter=int, default=32)  # Census builds older than this are deleted
    #
    # Block list of dataset IDs
    dataset_id_blocklist_uri: str = field(
        default="https://raw.githubusercontent.com/chanzuckerberg/cellxgene-census/main/tools/cellxgene_census_builder/dataset_blocklist.txt"
    )
    # User Agent header for all dataset requests from the datasets.cellxgene.cziscience.com route.
    user_agent_prefix: str = field(default="census-builder-")
    user_agent_environment: str = field(default="unknown")
    #
    # For testing convenience only
    manifest: str = field(default=None)
    test_first_n: int = field(converter=int, default=0)

    @classmethod
    def load(cls, file: Union[str, os.PathLike[str], io.TextIOBase]) -> Self:
        if isinstance(file, (str, os.PathLike)):
            with open(file) as f:
                user_config = yaml.safe_load(f)
        else:
            user_config = yaml.safe_load(file)

        # Empty YAML config file is legal
        if user_config is None:
            user_config = {}

        # But we only understand a top-level dictionary (e.g., no lists, etc.)
        if not isinstance(user_config, dict):
            raise TypeError("YAML config file malformed - expected top-level dictionary")

        return cls(**user_config)

    @classmethod
    def load_from_env_vars(cls) -> Self:
        config = cls()
        for fld in fields(CensusBuildConfig):
            fld_env_var = f"CENSUS_BUILD_{fld.name.upper()}"
            if fld_env_var in os.environ:
                setattr(config, fld.name, os.environ[fld_env_var])
        return config


class Namespace(Mapping[str, Any]):
    """Readonly namespace"""

    def __init__(self, **kwargs: Any):
        self._state = dict(kwargs)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Namespace):
            return self._state == other._state
        return NotImplemented

    def __contains__(self, key: Any) -> bool:
        return key in self._state

    def __repr__(self) -> str:
        items = (f"{k}={v!r}" for k, v in self.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __getitem__(self, key: str) -> Any:
        return self._state[key]

    def __getattr__(self, key: str) -> Any:
        return self._state[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._state)

    def __len__(self) -> int:
        return len(self._state)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class MutableNamespace(Namespace):
    """Mutable namespace"""

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError
        self._state[key] = value

    # Do not implement __delitem__. Log format has no deletion marker, so delete
    # semantics can't be supported until that is implemented.


class CensusBuildState(MutableNamespace):
    def __init__(self, **kwargs: Any):
        self.__dirty_keys = set(kwargs)
        super().__init__(**kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        if self._state.get(key) == value:
            return
        super().__setitem__(key, value)
        self.__dirty_keys.add(key)

    @classmethod
    def load(cls, file: Union[str, os.PathLike[str], io.TextIOBase]) -> Self:
        if isinstance(file, (str, os.PathLike)):
            with open(file) as state_log:
                documents = list(yaml.safe_load_all(state_log))
        else:
            documents = list(yaml.safe_load_all(file))

        state = cls(**functools.reduce(lambda acc, r: acc.update(r) or acc, documents, {}))
        state.__dirty_keys.clear()
        return state

    def commit(self, file: Union[str, os.PathLike[str]]) -> None:
        # append dirty elements (atomic on Posix)
        if self.__dirty_keys:
            dirty = {k: self[k] for k in self.__dirty_keys}
            self.__dirty_keys.clear()
            with open(file, mode="a") as state_log:
                record = f"--- # {datetime.now().isoformat()}\n" + yaml.dump(dirty)
                state_log.write(record)


@define(frozen=True)
class CensusBuildArgs:
    working_dir: pathlib.PosixPath = field(validator=validators.instance_of(pathlib.PosixPath))
    config: CensusBuildConfig = field(validator=validators.instance_of(CensusBuildConfig))
    state: CensusBuildState = field(
        factory=CensusBuildState, validator=validators.instance_of(CensusBuildState)  # default: empty state
    )

    @property
    def soma_path(self) -> pathlib.PosixPath:
        return self.working_dir / self.build_tag / "soma"

    @property
    def h5ads_path(self) -> pathlib.PosixPath:
        return self.working_dir / self.build_tag / "h5ads"

    @property
    def build_tag(self) -> str:
        build_tag = self.config.build_tag
        if not isinstance(build_tag, str):
            raise TypeError("Configuration contains non-string build_tag.")
        return build_tag
