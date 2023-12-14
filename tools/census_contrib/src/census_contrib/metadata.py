from __future__ import annotations

import datetime
import pathlib
from typing import Any, Dict, Optional, Tuple, Union, cast

import attrs
import cattrs
import cattrs.preconf.json
import cattrs.preconf.pyyaml
import cellxgene_census
import requests
from attrs import field, validators
from typing_extensions import Self

from .args import Arguments
from .census_util import open_census
from .util import get_logger

logger = get_logger()


def none_or_str(v: Optional[str]) -> str:
    return "" if v is None else v


@attrs.define(kw_only=True, frozen=True)
class Contact:
    name: str = field(validator=validators.instance_of(str))
    email: str = field(validator=validators.instance_of(str))
    affiliation: str = field(validator=validators.instance_of(str))


@attrs.define(kw_only=True, frozen=True)
class EmbeddingMetadata:
    id: str = field(validator=validators.instance_of(str))
    title: str = field(validator=validators.instance_of(str))
    description: str = field(validator=validators.instance_of(str))
    primary_contact: Contact = field(validator=validators.instance_of(Contact))
    additional_contacts: Tuple[Contact, ...] = field(
        factory=tuple,
        validator=validators.deep_iterable(
            validators.instance_of(Contact),
            validators.instance_of(tuple),
        ),
    )
    DOI: str = field(default="", converter=none_or_str, validator=validators.instance_of(str))
    additional_information: str = field(default="", converter=none_or_str, validator=validators.instance_of(str))
    model_link: str = field(default="", converter=none_or_str, validator=validators.instance_of(str))
    data_type: str = field(validator=validators.in_(("obs_embedding",)))
    census_version: str = field(validator=validators.instance_of(str))
    experiment_name: str = field(validator=validators.instance_of(str))
    measurement_name: str = field(validator=validators.instance_of(str))
    n_embeddings: int = field(validator=validators.instance_of(int))
    n_features: int = field(validator=validators.instance_of(int))
    submission_date: datetime.date = field(validator=validators.instance_of(datetime.date))

    @classmethod
    def from_dict(cls, md: Dict[str, Any]) -> Self:
        return cast(Self, cattrs.structure_attrs_fromdict(md, cls))

    @classmethod
    def from_yaml(cls, data: str) -> Self:
        return cast(
            Self,
            cattrs.preconf.pyyaml.make_converter(forbid_extra_keys=True, prefer_attrib_converters=True).loads(
                data, cls
            ),
        )

    @classmethod
    def from_json(cls, data: str) -> Self:
        return cast(
            Self,
            cattrs.preconf.json.make_converter(forbid_extra_keys=True, prefer_attrib_converters=True).loads(data, cls),
        )

    def to_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], cattrs.unstructure(self))

    def to_json(self) -> str:
        return cast(str, cattrs.preconf.json.make_converter().dumps(self))

    def to_yaml(self) -> str:
        return cast(str, cattrs.preconf.pyyaml.make_converter().dumps(self))


def load_metadata(path: Union[str, pathlib.Path]) -> EmbeddingMetadata:
    metadata_path = pathlib.PosixPath(path)
    if not metadata_path.is_file():
        raise ValueError("--metadata: file does not exist")

    with open(metadata_path) as f:
        data = f.read()

    try:
        if metadata_path.suffix in [".yml", ".yaml"]:
            cmd = EmbeddingMetadata.from_yaml(data)
        elif metadata_path.suffix == ".json":
            cmd = EmbeddingMetadata.from_json(data)
        else:
            raise ValueError("--metadata: unrecognized file format")
    except (ValueError, TypeError, cattrs.ClassValidationError) as e:
        raise ValueError(f"--metadata format error: {str(e)}") from e
    except cattrs.ForbiddenExtraKeysError as e:
        raise ValueError(f"metadata contained extra keys: {str(e)}") from e

    return cmd


def validate_metadata(args: Arguments, metadata: EmbeddingMetadata) -> EmbeddingMetadata:
    """
    Checks to perform on metadata:
    1. Census version must be an LTS version (implies existence)
    2. Census version, experiment and measurement must exist
    3. DOI must validate
    4. All supplied URLs must resolve
    5. Title must have length < 128 characters
    6. Description must have length < 2048 characters
    """

    if not metadata.id:
        raise ValueError("metadata is missing 'id' (accession)")

    validate_census_info(args, metadata)
    validate_doi(metadata)
    validate_urls(metadata)

    # 5. Title must have length < 128 characters
    MAX_TITLE_LENGTH = 128
    if not metadata.title or len(metadata.title) > MAX_TITLE_LENGTH:
        raise ValueError(
            f"Metadata: title must be string between 1 and {MAX_TITLE_LENGTH} characters in length",
        )

    # 6. Description must have length < 2048 characters
    MAX_DESCRIPTION_LENGTH = 2048
    if not metadata.description or len(metadata.description) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(
            "Metadata: description must be string between 1 and {MAX_DESCRIPTION_LENGTH} characters in length",
        )

    return metadata


def validate_census_info(args: Arguments, metadata: EmbeddingMetadata) -> None:
    """Errors / exists upon failure"""

    if not args.census_uri:  # ie. if no override of census
        releases = cellxgene_census.get_census_version_directory()

        # 1. Census version must exist
        if metadata.census_version not in releases:
            raise ValueError("Metadata specifies a census_version that does not exist.")

    # TODO - test for LTS?

    # 2. Census version, experiment and measurement must exist
    with open_census(census_version=metadata.census_version, census_uri=args.census_uri) as census:
        if metadata.experiment_name not in census["census_data"]:
            raise ValueError("Metadata specifies non-existent experiment_name")

        if metadata.measurement_name not in census["census_data"][metadata.experiment_name].ms:
            raise ValueError("Metadata specifies non-existent measurement_name")

        assert census["census_data"][metadata.experiment_name].obs.count > 0
        assert census["census_data"][metadata.experiment_name].ms[metadata.measurement_name].var.count > 0

        if metadata.n_embeddings > census["census_data"][metadata.experiment_name].obs.count:
            raise ValueError("Metadata n_embeddings larger than obs joinid")


def validate_doi(metadata: EmbeddingMetadata) -> None:
    """Errors / exists upon failure"""

    # 3. DOI must validate if specified
    if not metadata.DOI:
        return

    # doi.org returns 302 redirect for existing DOIs, 404
    # Assume that a 302 means a legit DOI
    url = f"https://doi.org/{metadata.DOI}"
    r = requests.get(url, allow_redirects=False)
    if r.status_code == 302 and "location" in r.headers:
        return

    raise ValueError("Metadata contains a DOI that does not resolve")


def validate_urls(metadata: EmbeddingMetadata) -> None:
    """Errors / exits upon failure"""

    # 4. All supplied URLs must resolve
    for fld_name, url in [(f, getattr(metadata, f, "")) for f in ("model_link",)]:
        if url:
            if url.startswith("https://"):
                r = requests.head(url, allow_redirects=True)
                if r.status_code == 200:
                    continue
                raise ValueError(f"Metadata contains unresolvable URL {fld_name}={url}")

            logger.warning(f"Unable to verify URI {fld_name}={url}")
