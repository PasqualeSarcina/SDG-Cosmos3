"""Wrap JSON captions in numbered Cosmos text-to-image input files."""

import argparse
import json
from pathlib import Path


def read_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        # Keep the whole structured caption, as in t2i.py.
        return json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True, help="Caption directory (searched recursively)")
    parser.add_argument("--negative-prompt", type=Path, required=True, help="Negative prompt JSON or TXT file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Destination directory for scene_XXXXXX.json")
    args = parser.parse_args()

    try:
        if not args.input.is_dir():
            raise ValueError(f"Caption directory does not exist: {args.input}")
        if args.negative_prompt.suffix.lower() not in (".json", ".txt"):
            raise ValueError("The negative prompt must be a JSON or TXT file")
        output = args.output.resolve()
        if args.input.resolve().is_relative_to(output):
            raise ValueError("Output must not be the input directory or one of its parents")
        negative_path = args.negative_prompt.resolve()
        captions = sorted(
            path for path in args.input.rglob("*")
            if path.is_file() and path.suffix.lower() == ".json"
            and path.resolve() != negative_path
            and not path.resolve().is_relative_to(output)
        )
        if not captions:
            raise ValueError(f"No caption JSON files found in {args.input}")
        negative_prompt = read_prompt(args.negative_prompt)
        # Validate all inputs and destination names before writing anything.
        scenes = []
        for index, caption in enumerate(captions, start=1):
            target = output / f"scene_{index:06d}.json"
            if target.exists():
                raise ValueError(f"Output already exists: {target}")
            scenes.append((target, {
                "model_mode": "text2image",
                "prompt": read_prompt(caption),
                "negative_prompt": negative_prompt,
                "resolution": "720",
                "num_frames": 1,
                "seed": 2026,
            }))
        output.mkdir(parents=True, exist_ok=True)
        for target, scene in scenes:
            with target.open("x", encoding="utf-8") as stream:
                json.dump(scene, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {len(scenes)} scenes to {output}")


if __name__ == "__main__":
    main()
