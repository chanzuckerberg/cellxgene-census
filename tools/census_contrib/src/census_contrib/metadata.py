from __future__ import annotations

import datetime
import json
import pathlib
import sys
from typing import TYPE_CHECKING, Any, Optional

import attrs
import cellxgene_census
import requests
import yaml
from attrs import field, validators

if TYPE_CHECKING:
    from .args import Arguments
from .util import error


def none_or_str(v: Optional[str]) -> str:
    return "" if v is None else v


@attrs.define(kw_only=True)
class ContribMetadata:
    title: str = field(validator=validators.instance_of(str))
    description: str = field(validator=validators.instance_of(str))
    contact_name: str = field(validator=validators.instance_of(str))
    contact_email: str = field(validator=validators.instance_of(str))
    contact_affiliation: str = field(validator=validators.instance_of(str))
    # DOI: str = field(default="", converter=str, validator=validators.instance_of(str))
    DOI: str = field(default="", converter=none_or_str, validator=validators.instance_of(str))
    additional_information: str = field(default="", converter=none_or_str, validator=validators.instance_of(str))
    model_link: str = field(default="", converter=none_or_str, validator=validators.instance_of(str))
    data_type: str = field(validator=validators.in_(("obs_embedding",)))
    census_version: str = field(validator=validators.instance_of(str))
    experiment_name: str = field(validator=validators.instance_of(str))
    measurement_name: str = field(validator=validators.instance_of(str))
    n_features: int = field(validator=validators.instance_of(int))
    submission_date: str = field(validator=validators.instance_of(str))

    @submission_date.validator
    def check(self, _: attrs.Attribute[Any], value: Any) -> None:
        try:
            datetime.date.fromisoformat(value)
        except ValueError as e:
            raise ValueError(f"submission_date not ISO date: expected 'YYYY-MM-DD', got '{value}'") from e

    def as_json(self) -> str:
        return json.dumps(attrs.asdict(self))


def load_metadata(args: "Arguments") -> ContribMetadata:
    metadata_path = pathlib.PosixPath(args.metadata)
    if not metadata_path.is_file():
        error(args, "--metadata: file does not exist")

    if metadata_path.suffix in [".yml", ".yaml"]:
        with open(metadata_path) as f:
            md = yaml.load(f, Loader=NoDatesSafeLoader)

    elif metadata_path.suffix == ".json":
        with open(metadata_path) as f:
            md = json.load(f)

    else:
        error(args, "--metadata: unrecognized file format")

    if not isinstance(md, dict):
        error(args, "--metadata: file format did not contain a dictionary")
    expected_fields = set(attrs.fields_dict(ContribMetadata).keys())
    found_fields = set(md.keys())
    if expected_fields ^ found_fields:
        print("metadata - unexpected fields.", file=sys.stderr)
        if found_fields - expected_fields:
            print(f"Extra: {found_fields - expected_fields}.", file=sys.stderr)
        if expected_fields - found_fields:
            print(f"Missing: {expected_fields - found_fields}", file=sys.stderr)
        error(args, "--metadata: unexpected metadata file contents")

    try:
        cmd = ContribMetadata(**md)
    except (ValueError, TypeError) as e:
        error(args, f"--metadata format error: {str(e)}")

    return cmd


# Acknowledgement: https://stackoverflow.com/a/37958106
class NoDatesSafeLoader(yaml.SafeLoader):
    @classmethod
    def remove_implicit_resolver(cls, tag_to_remove: str) -> None:
        """
        Remove implicit resolvers for a particular tag

        Takes care not to modify resolvers in super classes.

        We want to load datetimes as strings, not dates, because we
        go on to serialize as json which doesn't have the advanced types
        of yaml, and leads to incompatibilities down the track.
        """
        if "yaml_implicit_resolvers" not in cls.__dict__:
            cls.yaml_implicit_resolvers = cls.yaml_implicit_resolvers.copy()

        for first_letter, mappings in cls.yaml_implicit_resolvers.items():
            cls.yaml_implicit_resolvers[first_letter] = [
                (tag, regexp) for tag, regexp in mappings if tag != tag_to_remove
            ]


NoDatesSafeLoader.remove_implicit_resolver("tag:yaml.org,2002:timestamp")


def validate_metadata(args: "Arguments", metadata: ContribMetadata) -> None:
    """
    Checks to perform on metadata:
    1. Census version must be an LTS version (implies existence)
    2. Census version, experiment and measurement must exist
    3. DOI must validate
    4. All supplied URLs must resolve
    5. Title must have length < 96 characters
    6. Description must have length < 2048 characters
    """

    validate_census_info(args, metadata)
    validate_doi(args, metadata)
    validate_urls(args, metadata)

    # 5. Title must have length < 96 characters
    MAX_TITLE_LENGTH = 96
    if not metadata.title or len(metadata.title) > MAX_TITLE_LENGTH:
        error(
            args,
            "Metadata: title must be string between 1 and {MAX_TITLE_LENGTH} characters in length",
        )

    # 6. Description must have length < 2048 characters
    MAX_DESCRIPTION_LENGTH = 2048
    if not metadata.description or len(metadata.description) > MAX_DESCRIPTION_LENGTH:
        error(
            args,
            "Metadata: description must be string between 1 and {MAX_DESCRIPTION_LENGTH} characters in length",
        )


def validate_census_info(args: "Arguments", metadata: ContribMetadata) -> None:
    """Errors / exists upon failure"""
    lts_releases = cellxgene_census.get_census_version_directory(lts=True)

    # 1. Census version must be an LTS version (implies existence)
    if metadata.census_version in lts_releases:
        error(args, "Metadata specifies a census_version that is not an LTS release.")

    # 2. Census version, experiment and measurement must exist
    with cellxgene_census.open_soma(census_version=metadata.census_version) as census:
        if metadata.experiment_name not in census["census_data"]:
            error(args, "Metadata specifies non-existent experiment_name")

        if metadata.measurement_name not in census["census_data"][metadata.experiment_name].ms:
            error(args, "Metadata specifies non-existent measurement_name")

        assert census["census_data"][metadata.experiment_name].obs.count > 0
        assert census["census_data"][metadata.experiment_name].ms[metadata.measurement_name].var.count > 0


def validate_doi(args: "Arguments", metadata: ContribMetadata) -> None:
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

    error(args, "Metadata contains a DOI that does not resolve")


def validate_urls(args: "Arguments", metadata: ContribMetadata) -> None:
    """Errors / exits upon failure"""

    # 4. All supplied URLs must resolve
    for fld_name, url in [(f, getattr(metadata, f, "")) for f in ("additional_information", "model_link")]:
        if not url:
            continue

        r = requests.head(url, allow_redirects=True)
        if r.status_code == 200:
            continue

        error(args, f"Metadata contains unresolvable URL {fld_name}={url}")
