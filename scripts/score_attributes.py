"""Score field attributes on the original JSONs selected in representatives.json.

One LLM request per representative/field/attribute. Every representative has
equal weight; visibility is existential over cars, with one vote per scene.
Separately, cars_count frequencies are computed over all original JSON files.
Only complete, validated runs are saved as judgments.json and scores.json.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(_PACKAGE_DIR))

from src.request.client import chat_completion, create_client, resolve_model, run_batch, with_retries
from src.request.response_errors import response_content
from src.request.response_parsing import parse_json_response

log = logging.getLogger(__name__)
ATTRIBUTE_FIELDS = ("foreground", "background", "lighting", "weather", "visibility")
JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {"judgment": {"type": "integer", "enum": [0, 1]}},
    "required": ["judgment"],
    "additionalProperties": False,
}
VISIBILITY_UNIT = "Does at least one car in the scene have this characteristic?"


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to read {path}: {exc}") from exc


def cars_count_distribution(input_dir: Path) -> dict:
    """Count strictly positive JSON integers over all originals, without an LLM."""
    frequencies = Counter()
    exclusions = {
        "invalid JSON": 0,
        "content is not an object": 0,
        "missing field": 0,
        "invalid type": 0,
        "count below 1": 0,
    }
    paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )

    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
        except OSError as exc:
            raise ValueError(f"Unable to read {path}: {exc}") from exc
        except ValueError:
            reason = "invalid JSON"
        else:
            if not isinstance(data, dict):
                reason = "content is not an object"
            elif "cars_count" not in data:
                reason = "missing field"
            elif type(data["cars_count"]) is not int:
                # bool is a subclass of int: require the exact JSON integer type.
                reason = "invalid type"
            elif data["cars_count"] < 1:
                reason = "count below 1"
            else:
                frequencies[data["cars_count"]] += 1
                continue
        exclusions[reason] += 1
        log.debug("cars_count: excluded %s (%s)", path.name, reason)

    total = sum(frequencies.values())
    excluded = sum(exclusions.values())
    log.info("cars_count: %d JSON files examined, %d valid scenes, %d files excluded", len(paths), total, excluded)
    if excluded:
        log.warning(
            "cars_count: %d files excluded: %s", excluded,
            "; ".join(f"{reason}: {count}" for reason, count in exclusions.items()),
        )
    if not total:
        raise ValueError(
            f"No valid cars_count values in {input_dir}: expected an integer >= 1, "
            f"excluding booleans. {excluded} files excluded out of {len(paths)} JSON files."
        )
    return {
        "total_valid_scenes": total,
        "frequencies": {
            str(value): {"count": frequencies[value], "probability": frequencies[value] / total}
            for value in sorted(frequencies)
        },
    }


def load_attributes(path: Path) -> dict[str, list[str]]:
    data = read_json(path)
    if not isinstance(data, dict) or set(data) != set(ATTRIBUTE_FIELDS):
        raise ValueError(f"{path}: expected exactly these fields: {', '.join(ATTRIBUTE_FIELDS)}.")
    for field, values in data.items():
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"{path}: {field} must be a list of nonempty strings.")
        # Score each distinct attribute once, retaining its original name/order.
        data[field] = list(dict.fromkeys(values))
    if not any(data.values()):
        raise ValueError(f"{path}: no attributes to evaluate.")
    return data


def load_representatives(path: Path, input_dir: Path) -> list[dict]:
    """Load only selected originals and validate all of them before any requests."""
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a nonempty list of representatives with an id.")
    examples, seen = [], set()
    for record in records:
        scene_id = record.get("id") if isinstance(record, dict) else None
        if (
            not isinstance(scene_id, str) or not scene_id.strip()
            or scene_id in (".", "..") or any(char in scene_id for char in '/\\:')
        ):
            raise ValueError(f"{path}: invalid representative id: {scene_id!r}.")
        if scene_id in seen:
            raise ValueError(f"{path}: duplicate representative id: {scene_id!r}.")
        seen.add(scene_id)
        source = (input_dir / f"{scene_id}.json").resolve()
        if source.parent != input_dir.resolve():
            raise ValueError(f"Original JSON file is outside the input directory: {source}.")
        data = read_json(source)
        if not isinstance(data, dict):
            raise ValueError(f"{source}: expected a JSON object.")
        fields = {}
        for field in ATTRIBUTE_FIELDS[:-1]:
            value = data.get(field)
            if not isinstance(value, str):
                raise ValueError(f"{source}: {field} must be present and must be a string.")
            fields[field] = value
        cars = data.get("cars")
        if not isinstance(cars, list) or any(
            not isinstance(car, dict) or not isinstance(car.get("visibility"), str)
            for car in cars
        ):
            raise ValueError(f"{source}: cars must be a list of objects with a string visibility field.")
        fields["visibility"] = [car["visibility"] for car in cars]
        examples.append({"id": scene_id, "source_file": str(source), "fields": fields})
    return examples


async def request_judgment(args, client, model, system_prompt, job, index) -> int:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({
            "field": job["field"], "content": job["content"], "attribute": job["attribute"],
        }, ensure_ascii=False)},
    ]

    async def attempt(attempt_index):
        response_path = (
            args.output / "debug" / f"judgment-{index:07d}.attempt-{attempt_index}.json"
            if args.debug else None
        )
        response = await chat_completion(
            client, response_path=response_path, model=model, messages=messages,
            max_tokens=args.max_tokens, temperature=1.0, top_p=0.95,
            presence_penalty=0.0, reasoning_effort=args.reasoning_effort,
            response_format={
                "type": "json_schema", "json_schema": {
                    "name": "attribute_judgment", "strict": True, "schema": JUDGMENT_SCHEMA,
                },
            },
            extra_body={
                "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        text = response_content(response, max_tokens=args.max_tokens, response_path=response_path)
        text = re.sub(r"\A\s*<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
        data = parse_json_response(text)
        if (
            not isinstance(data, dict) or set(data) != {"judgment"}
            or type(data["judgment"]) is not int or data["judgment"] not in (0, 1)
        ):
            raise ValueError('Expected exactly {"judgment": 0} or {"judgment": 1}.')
        return data["judgment"]

    return await with_retries(
        f"Score {job['representative_id']}/{job['field']}/{job['attribute']}", attempt,
        max_attempts=args.max_retries, delay_s=1.0, server=args.server,
    )


async def score_attributes(args, system_prompt, examples, attributes, count_distribution) -> dict:
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1.")
    if not examples or not any(attributes.values()):
        raise ValueError("Representatives and attributes are required for evaluation.")
    jobs = [
        {
            "representative_id": example["id"], "source_file": example["source_file"],
            "field": field, "attribute": attribute, "content": example["fields"][field],
        }
        for example in examples
        for field in ATTRIBUTE_FIELDS
        for attribute in attributes[field]
    ]
    async with create_client(args.server, api_token=args.api_token, timeout_s=args.timeout) as client:
        model = await resolve_model(client, args.model)
        log.info("Using model: %s; %d representatives; %d judgments", model, len(examples), len(jobs))

        async def process(index):
            value = await request_judgment(args, client, model, system_prompt, jobs[index], index)
            return {**jobs[index], "judgment": value}

        judgments = await run_batch(
            range(len(jobs)), process, max_workers=args.max_workers,
            desc="Scoring attributes", unit="judgment",
        )

    totals = {(field, attribute): [0, 0] for field in ATTRIBUTE_FIELDS for attribute in attributes[field]}
    for row in judgments:
        counts = totals[row["field"], row["attribute"]]
        counts[0] += row["judgment"]
        counts[1] += 1
    scores = {field: [] for field in ATTRIBUTE_FIELDS}
    for (field, attribute), (positives, evaluated) in totals.items():
        scores[field].append({
            "attribute": attribute, "positives": positives,
            "texts_evaluated": evaluated, "score": positives / evaluated,
        })
    result = {
        "model": model, "representatives_count": len(examples),
        "score_definition": "positives / texts_evaluated; equal weight per representative",
        "visibility_unit": VISIBILITY_UNIT, "scores": scores,
        "cars_count_distribution": count_distribution,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("judgments.json", judgments), ("scores.json", result)):
        (args.output / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        log.info("Saved: %s", args.output / filename)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing the original scene JSONs.")
    parser.add_argument("--representatives", type=Path, required=True, help="representatives.json from select_representatives.py.")
    parser.add_argument("--attributes", type=Path, required=True, help="attributes.json from discover_attributes.py.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for judgments.json and scores.json.")
    parser.add_argument("--server", required=True, help="OpenAI-compatible endpoint URL.")
    parser.add_argument("--model", default=None, help="Defaults to the first model returned by the endpoint.")
    parser.add_argument("--api-token", "--api_token", default=None)
    parser.add_argument("--max-workers", "--max_workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16384, help="Output budget, including thinking.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum attempts, including the first.")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    parser.add_argument("--debug", action="store_true", help="Save raw API responses under output/debug/.")
    args = parser.parse_args()
    for name in ("max_workers", "max_tokens", "max_retries"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1.")
    if not args.timeout > 0:
        parser.error("--timeout must be > 0.")
    if not args.server.strip() or (args.model is not None and not args.model.strip()):
        parser.error("--server and --model cannot be empty.")
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if not args.input.is_dir():
        parser.error("Input directory not found.")
    if args.output.exists() and not args.output.is_dir():
        parser.error("Output path must be a directory.")
    if args.output == args.input:
        parser.error("Use a separate output directory.")
    protected = [args.attributes.resolve(), args.representatives.resolve()]
    if any(args.output / name in protected for name in ("judgments.json", "scores.json")):
        parser.error("Output files must not overwrite input files.")
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        count_distribution = cars_count_distribution(args.input)
        attributes = load_attributes(args.attributes)
        examples = load_representatives(args.representatives, args.input)
    except ValueError as exc:
        parser.error(str(exc))
    system_prompt = (_PACKAGE_DIR / "src/templates/score_attributes.txt").read_text(encoding="utf-8")
    asyncio.run(score_attributes(args, system_prompt, examples, attributes, count_distribution))


if __name__ == "__main__":
    main()
