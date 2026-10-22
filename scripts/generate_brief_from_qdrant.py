"""Generate compact scene JSONs from background categories and Qdrant references."""

import asyncio
import copy
import json
import logging
import math
import os
import random
import re
import sys
import threading
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from jsonschema import Draft202012Validator
from tqdm.contrib.logging import logging_redirect_tqdm

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from src.embeddings import MODEL_NAME, encode_documents, load_encoder
from src.request.client import chat_completion, create_client, resolve_model, run_batch, with_retries
from src.request.response_errors import response_content

log = logging.getLogger(__name__)
SCHEMA_PATH = ROOT / "src/schema/description_json_schema.json"
PROMPT_DIR = ROOT / "src/templates"
TEXT_FIELDS = ("foreground", "background", "lighting", "weather")
CAR_SIZE_DISTRIBUTION = {"small": 0.217, "medium": 0.485, "large": 0.298}


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    text = re.sub(r"\A\s*<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL).strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"\A```(?:json)?\s*", "", text, count=1).removesuffix("```").strip()
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_categories(path):
    values = strict_json(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(values, list) or not values:
        raise ValueError("--categories must contain a nonempty JSON array of strings.")
    categories, seen = [], set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Every background category must be a nonempty string.")
        name = " ".join(value.split())
        if name.casefold() not in seen:
            seen.add(name.casefold())
            categories.append(name)
    return categories


def brief_schema():
    """Extend the caption schema in memory; do not change the shared source file."""
    schema = copy.deepcopy(strict_json(SCHEMA_PATH.read_text(encoding="utf-8-sig")))
    if set(schema["properties"]) != set(TEXT_FIELDS):
        raise ValueError("The brief schema must define foreground, background, lighting and weather.")
    for name in TEXT_FIELDS:
        schema["properties"][name]["minLength"] = 1
    schema["properties"]["cars_count"] = {"type": "integer", "minimum": 1}
    schema["required"] = [*TEXT_FIELDS, "cars_count"]
    schema["additionalProperties"] = False
    Draft202012Validator.check_schema(schema)
    for name in ("lighting", "weather"):
        values = schema["properties"][name].get("enum")
        if not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{name} must define nonempty string enum values.")
    return schema


@dataclass
class Args:
    categories: Path
    """background_categories.json produced by discover_background.py."""
    output: Path
    """New or empty directory for the N compact scene JSONs."""
    server: str
    """OpenAI-compatible LLM endpoint; model is discovered automatically."""
    number: int = 1
    """Number of scenes to attempt."""
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection: str = "description"
    api_token: str | None = None
    seed: int = 2026
    max_workers: int = 8
    top_k: int = 512
    """Maximum references retrieved per search and sent in each LLM prompt."""
    score_threshold: float | None = None
    """Minimum Qdrant similarity score; omit to keep all top-k results."""
    max_tokens: int = 16384
    max_retries: int = 3
    timeout: float = 3600.0
    reasoning_effort: Literal["low", "medium", "xhigh"] = "medium"
    device: Literal["auto", "cuda", "cpu"] = "auto"

    def __post_init__(self):
        for name in ("number", "max_workers", "top_k", "max_tokens", "max_retries"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be an integer >= 1.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("--timeout must be positive and finite.")
        if self.score_threshold is not None and not math.isfinite(self.score_threshold):
            raise ValueError("--score-threshold must be finite.")
        for name in ("server", "url", "collection"):
            if not getattr(self, name).strip():
                raise ValueError(f"--{name.replace('_', '-')} cannot be empty.")
        if self.reasoning_effort not in ("low", "medium", "xhigh"):
            raise ValueError("Invalid reasoning effort.")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("Invalid embedding device.")
        self.categories = Path(self.categories).resolve()
        self.output = Path(self.output).resolve()
        read_categories(self.categories)
        for directory in (self.output, self.trace_output):
            if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
                raise ValueError(f"Use a new or empty directory: {directory}")
            if self.categories.is_relative_to(directory) or SCHEMA_PATH.is_relative_to(directory):
                raise ValueError("Output directories cannot contain the input categories or schema.")

    @property
    def trace_output(self):
        return self.output.with_name(self.output.name + "_logs")


def load_distribution(client, collection):
    """Ask Qdrant for exact cars_count frequencies, without downloading records."""
    limit = 256
    while True:
        facets = client.facet(collection_name=collection, key="cars_count", exact=True, limit=limit)
        if len(facets.hits) < limit:
            break
        # A full response may omit rarer values; expand until every value is returned.
        limit *= 2
    histogram, total, adjusted = Counter(), 0, 0
    for hit in facets.hits:
        count = hit.value
        # Treat zero as one car so every record contributes to car-containing scenes.
        histogram[1 if count == 0 else count] += hit.count
        total += hit.count
        if count == 0:
            adjusted += hit.count
    if not histogram:
        raise ValueError(f"Collection {collection!r} has no records.")
    return dict(sorted(histogram.items())), total, adjusted


def sample_plans(args, categories, histogram, schema):
    """Give each scene an independent seed before concurrent work starts."""
    master = random.Random(args.seed)
    plans = []
    for index in range(args.number):
        seed = master.getrandbits(64)
        rng = random.Random(seed)
        plan = {
            "id": f"scene_{index + 1:06d}", "seed": seed,
            "macro_type": rng.choice(categories),
            "cars_count": rng.choices(list(histogram), weights=list(histogram.values()), k=1)[0],
            "lighting": rng.choice(schema["properties"]["lighting"]["enum"]),
            "weather": rng.choice(schema["properties"]["weather"]["enum"]),
        }
        # Sample each car's apparent size; small cars belong to the background.
        sizes = Counter(rng.choices(list(CAR_SIZE_DISTRIBUTION),
                                    weights=list(CAR_SIZE_DISTRIBUTION.values()), k=plan["cars_count"]))
        plan["car_size_counts"] = {size: sizes[size] for size in CAR_SIZE_DISTRIBUTION}
        plan["background_cars_count"] = sizes["small"]
        plan["foreground_cars_count"] = sizes["medium"] + sizes["large"]
        plans.append(plan)
    return plans


class Retriever:
    def __init__(self, client, encoder, args):
        self.client, self.encoder, self.args = client, encoder, args
        # One encoder is shared by concurrent scenes; serialize GPU and sync Qdrant work.
        self.lock = threading.Lock()

    def search(self, text, *, foreground=False):
        task = ("Retrieve foreground descriptions of subjects and activities compatible with the given setting."
                if foreground else "Retrieve descriptions of places matching the given background place type.")
        query = f"Instruct: {task}\nQuery: {text}"
        with self.lock:
            vector = encode_documents(self.encoder, [query], batch_size=1)[0].tolist()
            hits = self.client.query_points(
                collection_name=self.args.collection, query=vector,
                using="foreground" if foreground else "background",
                limit=self.args.top_k, with_vectors=False,
                score_threshold=self.args.score_threshold,
                with_payload=[*TEXT_FIELDS, "cars_count"] if foreground else ["background"],
            ).points
        records = []
        for hit in hits:
            payload = hit.payload or {}
            fields = TEXT_FIELDS if foreground else ("background",)
            for name in fields:
                if not isinstance(payload.get(name), str) or not payload[name].strip():
                    raise ValueError(f"Point {hit.id}: missing nonempty {name} text.")
            record = {"id": hit.id, "score": hit.score, **{name: payload[name] for name in fields}}
            if foreground:
                count = payload["cars_count"]
                # Apply the same zero-to-one conversion used for the distribution.
                record["cars_count"] = 1 if count == 0 else count
            records.append(record)
        if not records:
            raise ValueError(f"Qdrant returned no references (score_threshold={self.args.score_threshold}).")
        return records


def prepare_messages(args, prompt, fixed, candidates, rng):
    """Sample whole references by count, without limiting their text length."""
    references = rng.sample(candidates, len(candidates))[:args.top_k]
    if not references:
        raise ValueError("No references available.")
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": dump({**fixed, "references": references})}]
    return messages, references


async def request_stage(args, client, model, name, messages, trace):
    """Request one description as plain text, retrying empty or incomplete answers."""
    async def attempt(index):
        trace["attempts"] = index + 1
        response = await chat_completion(
            client, model=model, messages=messages, max_tokens=args.max_tokens,
            temperature=1.0, top_p=0.95, presence_penalty=0.0,
            reasoning_effort=args.reasoning_effort,
            extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                        "chat_template_kwargs": {"enable_thinking": True}},
        )
        text = response_content(response, max_tokens=args.max_tokens)
        text = re.sub(r"\A\s*<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL).strip()
        if not text:
            raise ValueError(f"{name} description cannot be empty.")
        return text

    return await with_retries(name, attempt, max_attempts=args.max_retries,
                              delay_s=1.0, server=args.server)


def write_json(path, data):
    """Publish complete JSONs through a temporary file, refusing to overwrite results."""
    temporary = path.with_suffix(".tmp")
    if path.exists():
        raise ValueError(f"Output already exists: {path}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def generate(args, qdrant, *, encoder=None):
    categories, schema = read_categories(args.categories), brief_schema()
    for directory in (args.output, args.trace_output):
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError(f"Use a new or empty directory: {directory}")
    histogram, total, adjusted = load_distribution(qdrant, args.collection)
    plans = sample_plans(args, categories, histogram, schema)
    prompts = {name: (PROMPT_DIR / f"generate_brief_{name}.txt").read_text(encoding="utf-8")
               for name in ("background", "foreground")}
    retriever = Retriever(qdrant, encoder if encoder is not None else load_encoder(args.device), args)
    for directory in (args.output, args.trace_output):
        directory.mkdir(parents=True, exist_ok=True)
    valid_total = sum(histogram.values())
    write_json(args.trace_output / "run.json", {
        "collection": args.collection, "categories": categories, "seed": args.seed,
        "embedding_model": MODEL_NAME, "foreground_vector": "foreground",
        "total_records": total, "zero_count_records_adjusted": adjusted,
        "cars_count_distribution": {str(count): {"frequency": frequency, "probability": frequency / valid_total}
                                    for count, frequency in histogram.items()},
        "car_size_distribution": CAR_SIZE_DISTRIBUTION,
        "number": args.number, "top_k": args.top_k, "score_threshold": args.score_threshold,
        "max_tokens": args.max_tokens,
        "max_retries": args.max_retries, "reasoning_effort": args.reasoning_effort,
    })
    async with create_client(args.server, api_token=args.api_token, timeout_s=args.timeout) as client:
        model = await resolve_model(client, None)

        async def scene(plan):
            trace = {"conditions": plan, "model": model, "status": "started", "stages": {}}
            rng = random.Random(plan["seed"])
            try:
                candidates = await asyncio.to_thread(retriever.search, plan["macro_type"])
                # Lighting/weather are deliberately absent: this phase fixes only the place.
                # Size classes stay in the plan; the LLM receives only the car counts.
                messages, refs = prepare_messages(args, prompts["background"], {
                    "macro_type": plan["macro_type"], "cars_count": plan["cars_count"],
                    "required_background_cars_count": plan["background_cars_count"],
                }, candidates, rng)
                stage = trace["stages"]["background"] = {"references": refs}
                background = await request_stage(args, client, model, "background", messages, stage)
                stage["result"] = background
                candidates = await asyncio.to_thread(retriever.search, background, foreground=True)
                remaining = plan["foreground_cars_count"]
                if remaining and not any(ref["cars_count"] > 0 for ref in candidates):
                    extra = await asyncio.to_thread(
                        retriever.search, f"Passenger cars parked or driving in this setting: {background}",
                        foreground=True,
                    )
                    candidates = list({ref["id"]: ref for ref in candidates + extra}.values())
                    trace["automotive_fallback_search"] = True
                messages, refs = prepare_messages(args, prompts["foreground"], {
                    "macro_type": plan["macro_type"], "background": background,
                    "background_cars_count": plan["background_cars_count"],
                    "lighting": plan["lighting"], "weather": plan["weather"],
                    "cars_count": plan["cars_count"], "required_foreground_cars_count": remaining,
                }, candidates, rng)
                stage = trace["stages"]["foreground"] = {"references": refs}

                foreground = await request_stage(args, client, model, "foreground", messages, stage)
                stage["result"] = foreground
                result = {"foreground": foreground, "background": background,
                          "lighting": plan["lighting"], "weather": plan["weather"],
                          "cars_count": plan["cars_count"]}
                Draft202012Validator(schema).validate(result)
                write_json(args.output / f"{plan['id']}.json", result)
                trace["status"] = "completed"
            except asyncio.CancelledError:
                trace["status"] = "interrupted"
                raise
            except Exception as exc:
                trace.update(status="failed", error=str(exc))
                log.error("%s failed: %s", plan["id"], exc)
            finally:
                write_json(args.trace_output / f"{plan['id']}.json", trace)
            return {"id": plan["id"], "status": trace["status"]}

        results = await run_batch(plans, scene, max_workers=args.max_workers,
                                  desc="Generating brief scenes", unit="scene")
    failures = [result["id"] for result in results if result["status"] != "completed"]
    return {"requested": args.number, "valid": len(results) - len(failures),
            "failed": len(failures), "failed_ids": failures, "output": str(args.output)}


def cli_main(argv=None):
    args = tyro.cli(Args, args=argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    for name in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from qdrant_client import QdrantClient

        with logging_redirect_tqdm(), closing(
            QdrantClient(url=args.url, api_key=os.getenv("QDRANT_API_KEY"), timeout=120)
        ) as qdrant:
            summary = asyncio.run(generate(args, qdrant))
    except Exception as exc:
        log.error("Generation aborted: %s", exc)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
