"""Shared token categories for preprocessing, collation and the Cosmos loss."""

from enum import IntEnum


class TokenClass(IntEnum):
    EXCLUDED = 0
    CONTENT = 1
    DECISION = 2
    STRUCTURE = 3
