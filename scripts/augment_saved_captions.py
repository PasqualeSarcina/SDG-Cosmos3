"""Apply only caption postprocessing to saved captions, without VLM requests."""

import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentation.augment_caption import augment_caption
from src.caption.structured_caption import assemble_caption_json


def augment_saved_captions(
    input_dir: Path, output_dir: Path, weather: str = "rainy", time_of_day: str = "day",
) -> int:
    input_dir = input_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output must be separate from the source caption tree")
    captions = sorted(input_dir.rglob("caption.json"))
    if not captions:
        raise ValueError(f"No caption.json files found in {input_dir}")

    # Validate the complete batch before creating any output files.
    prepared = []
    for path in captions:
        data = json.loads(path.read_text(encoding="utf-8"))
        phase2 = data.pop("comprehensive_t2i_caption")
        phase1, text = augment_caption(data, phase2, weather=weather, time_of_day=time_of_day)
        result = assemble_caption_json(phase1, text, {})
        destination = output_dir / path.parent.relative_to(input_dir)
        for filename in ("caption.json", "caption.txt"):
            if (destination / filename).exists():
                raise FileExistsError(destination / filename)
        prepared.append((destination, result, text))

    for destination, result, text in prepared:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "caption.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (destination / "caption.txt").write_text(text, encoding="utf-8")
        # Read back both files to verify the final saved format.
        saved = json.loads((destination / "caption.json").read_text(encoding="utf-8"))
        assert saved["comprehensive_t2i_caption"] == (destination / "caption.txt").read_text(encoding="utf-8")
        assert isinstance(saved["aesthetics"]["color_scheme"], str)
        assert "palette" not in saved["aesthetics"] and "intensity" not in saved["aesthetics"]
        print(destination)
    return len(prepared)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weather", default="rainy")
    parser.add_argument("--time-of-day", choices=("day", "night"), default="day")
    args = parser.parse_args()
    count = augment_saved_captions(args.input_dir, args.output_dir, args.weather, args.time_of_day)
    print(f"Saved {count} augmented captions using {args.weather}/{args.time_of_day}; no model requests.")
