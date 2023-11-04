from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .args import Arguments


def error(args: "Arguments", msg: str, status: int = 2) -> None:
    """Hard error, print message and exit with status"""
    print(f"{args.prog} - {msg}", file=sys.stderr)
    sys.exit(status)
