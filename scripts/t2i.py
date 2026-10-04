# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate images from a caption JSON file or a directory of caption.json files."""

import argparse
import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


def collect_inputs(input_path: Path, output_dir: Path, negative_prompt_path: Path|None = None) ->  tuple[list[tuple[Path, Path]], Path|None] :
    """Find captions and reject ambiguous output names before inference."""
    if negative_prompt_path is not None:
        if negative_prompt_path.is_file():
            if negative_prompt_path.suffix.lower() != ".json":
                raise ValueError(f"Expected a JSON file: {negative_prompt_path}")
        else:
            raise ValueError(f"Expected a JSON file: {negative_prompt_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Expected a JSON file: {input_path}")
        return [(input_path, output_dir / "image.png")], negative_prompt_path
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    captions = sorted(path for path in input_path.rglob("caption.json") if path.is_file())
    if not captions:
        raise ValueError(f"No caption.json files found in {input_path}")

    items = []
    names = set()
    for caption in captions:
        name = f"{caption.parent.name}.png"
        if name.casefold() in names:
            raise ValueError(f"Duplicate output name {name!r}: caption folders must have unique names")
        names.add(name.casefold())
        items.append((caption, output_dir / name))
    return items, negative_prompt_path


def generate_image(caption: Path, negative_prompt: Path|None, output: Path, server: str) -> None:
    prompt = json.dumps(json.loads(caption.read_text(encoding="utf-8-sig")), ensure_ascii=False, separators=(",", ":"))
    if negative_prompt is not None:
        negative_prompt = json.dumps(json.loads(negative_prompt.read_text(encoding="utf-8-sig")), ensure_ascii=False, separators=(",", ":"))
    request = {
        "model_mode": "text2image",
        "prompt": prompt,
        "negative_prompt": negative_prompt if negative_prompt is not None else "",
        "resolution": "720",
        "num_frames": 1,
        #"fps": 24.0,
        #"num_inference_steps": 50,
        #"guidance_scale": 4,
        #"flow_shift": 3.0,
        "seed": 2026,
    }
    # print(f"negative_prompt_chars={len(request['negative_prompt'])}")
    response = requests.post(f"{server.rstrip('/')}/v1/infer", json=request, timeout=1800)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        # requests' default message omits the server's validation/rejection reason.
        detail = response.text.strip()
        if detail:
            raise requests.HTTPError(f"{exc}\nServer response: {detail}", response=response) from exc
        raise
    encoded = response.json()["b64_image"]
    if not isinstance(encoded, str):
        raise ValueError("Expected b64_image to be a base64 string")
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.save(output, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Caption JSON file or directory searched recursively for caption.json",
    )
    parser.add_argument(
        "--negative-prompt", type=Path, required=False,
        help="Negative prompt JSON file",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory for PNG images")
    parser.add_argument(
        "--server", default="http://localhost:8000",
        help="Generator server URL without /v1 (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2,
        help="Maximum simultaneous requests (default: 4; use 1 for sequential requests)",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    try:
        items = collect_inputs(args.input, args.output, args.negative_prompt)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    negative_prompt_path = items[1]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(generate_image, caption, negative_prompt_path, output, args.server): (caption, output)
            for caption, output in items[0]
        }
        with tqdm(total=len(futures), desc="Generating images", unit="image", disable=not args.input.is_dir()) as progress:
            for future in as_completed(futures):
                caption, output = futures[future]
                progress.set_postfix_str(caption.parent.name)
                try:
                    future.result()
                except (OSError, ValueError, KeyError, TypeError, IndexError, requests.RequestException) as exc:
                    failures += 1
                    tqdm.write(f"Failed {caption}: {exc}")
                else:
                    tqdm.write(f"Saved image to {output}")
                progress.update(1)
    print(f"Saved {len(items[0]) - failures}/{len(items[0])} images to {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
