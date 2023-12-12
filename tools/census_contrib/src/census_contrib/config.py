from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from .args import Arguments
    from .metadata import EmbeddingMetadata


@attrs.define(kw_only=True, frozen=True)
class Config:
    args: "Arguments"
    metadata: "EmbeddingMetadata"
