"""Canonical output dimensions and JSON serialization for generated captions."""

import os
from dataclasses import dataclass
from typing import Any


# Cosmos was trained on captions serialized with json.dumps' ASCII default.
JSON_ENSURE_ASCII = bool(int(os.environ.get("JSON_ENSURE_ASCII", "1")))

RESOLUTION_RATIO_DICT: dict[str, dict[str, dict[str, int]]] = {
    "720": {
        "1,1": {"W": 960, "H": 960},
        "4,3": {"W": 1104, "H": 832},
        "3,4": {"W": 832, "H": 1104},
        "16,9": {"W": 1280, "H": 720},
        "9,16": {"W": 720, "H": 1280},
    },
}


@dataclass(frozen=True)
class OutputParameters:
    resolution: str
    aspect_ratio: str

    def __post_init__(self) -> None:
        if self.resolution not in RESOLUTION_RATIO_DICT:
            raise ValueError(f"Unsupported resolution {self.resolution!r}.")
        if self.aspect_ratio not in RESOLUTION_RATIO_DICT[self.resolution]:
            raise ValueError(
                f"Unsupported aspect_ratio {self.aspect_ratio!r} for resolution {self.resolution!r}."
            )


def apply_t2i_output_parameters(data: dict[str, Any], *, resolution: str, aspect_ratio: str) -> dict[str, Any]:
    """Set canonical image dimensions and aspect ratio on a T2I caption."""
    OutputParameters(resolution, aspect_ratio)
    pair = RESOLUTION_RATIO_DICT[resolution][aspect_ratio]
    data["resolution"] = {"H": pair["H"], "W": pair["W"]}
    data["aspect_ratio"] = aspect_ratio
    return data
