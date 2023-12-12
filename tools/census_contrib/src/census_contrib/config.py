from __future__ import annotations

import attrs
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .args import Arguments
    from .metadata import EmbeddingMetadata


@attrs.define(kw_only=True, frozen=True)
class Config:
    args: "Arguments"
    metadata: "EmbeddingMetadata"
