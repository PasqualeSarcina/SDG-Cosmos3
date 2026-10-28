"""Read XML captions, embed via API or local vLLM, and save them in Qdrant."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

import tyro
import numpy as np
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lora.dataset import parse_caption
from src.validation.scene_draft import validate_scene_draft

log = logging.getLogger(__name__)


@dataclass
class Args:
    input: list[Path] = field(default_factory=list)
    """XML files or directories to search recursively."""
    output_emb: Path | None = None
    """Export float32 .npz embeddings to this folder without connecting to Qdrant."""
    input_emb: Path | None = None
    """Import a .npz file or folder into Qdrant without loading an embedding model."""
    embedding_base_url: str | None = None
    """OpenAI-compatible URL including /v1, or EMBEDDING_BASE_URL."""
    embedding_backend: Literal["api", "vllm"] = "api"
    """Use an OpenAI-compatible API or the optional local vLLM Python library."""
    qdrant_url: str | None = None
    model: str = "Qwen/Qwen3-Embedding-4B"
    dimension: int = 2560
    """Expected embedding size: 2560 for 4B, 4096 for 8B."""
    scenes_collection: str = "caption_scenes"
    components_collection: str = "caption_components"
    batch_size: int = 16
    """Maximum texts per embedding request and points per Qdrant upload."""
    concurrency: int = 4
    """Maximum concurrent file workers and embedding requests."""
    timeout: float = 120.0
    vllm_max_model_len: int = 4096
    """Local model context limit; longer inputs are rejected, not truncated."""
    vllm_gpu_memory_utilization: float = 0.8
    vllm_enforce_eager: bool = False
    """Disable CUDA graphs in local vLLM to reduce their memory overhead."""

    def __post_init__(self):
        self.input = [Path(p) for p in self.input]
        self.output_emb = Path(self.output_emb) if self.output_emb is not None else None
        self.input_emb = Path(self.input_emb) if self.input_emb is not None else None
        if self.input_emb is not None and (self.input or self.output_emb is not None):
            raise ValueError("--input-emb cannot be combined with --input or --output-emb")
        if not self.input and self.input_emb is None:
            raise ValueError("Supply --input or --input-emb")
        for name in ("dimension", "batch_size", "concurrency"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if self.embedding_backend not in {"api", "vllm"}:
            raise ValueError("embedding_backend must be api or vllm")
        if self.embedding_backend == "vllm" and self.input_emb is None:
            if type(self.vllm_max_model_len) is not int or self.vllm_max_model_len < 1:
                raise ValueError("vllm_max_model_len must be a positive integer")
            if not 0 < self.vllm_gpu_memory_utilization < 1:
                raise ValueError("vllm_gpu_memory_utilization must be between 0 and 1 (exclusive)")
        if not all(s.strip() for s in (self.model, self.scenes_collection, self.components_collection)):
            raise ValueError("Model and collection names must not be empty")
        if self.scenes_collection == self.components_collection:
            raise ValueError("Scene and component collections must differ")
        self.embedding_base_url = self.embedding_base_url or os.getenv("EMBEDDING_BASE_URL")
        self.qdrant_url = self.qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        if self.input_emb is None and self.embedding_backend == "api" and not self.embedding_base_url:
            raise ValueError("Supply --embedding-base-url or EMBEDDING_BASE_URL")


def collect_paths(inputs: list[Path], suffix: str = ".xml") -> list[Path]:
    """Collect filenames only; each worker reads file contents later."""
    paths = {}
    for source in inputs:
        source = Path(source).resolve()
        if source.is_file():
            if source.suffix.lower() != suffix:
                expected = "an XML" if suffix == ".xml" else "an NPZ"
                raise ValueError(f"Expected {expected} file: {source}")
            candidates = [source]
        elif source.is_dir():
            candidates = (p for p in source.rglob("*") if p.is_file()
                          and p.suffix.lower() == suffix
                          and "debug" not in {part.lower() for part in p.relative_to(source).parts}
                          and not p.name.startswith("_"))
        else:
            raise FileNotFoundError(source)
        for path in candidates:
            path = path.resolve()
            paths[os.path.normcase(str(path))] = path
    if not paths:
        raise ValueError(f"No {suffix[1:].upper()} files found")
    return sorted(paths.values())


def embedding_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def has_content(value: Any) -> bool:
    if isinstance(value, dict):
        return any(has_content(v) for v in value.values())
    if isinstance(value, list):
        return any(has_content(v) for v in value)
    return isinstance(value, str) and value.strip().casefold() not in {"", "n/a", "none", "unknown"}


def read_record(path: Path) -> dict:
    """Read one Cosmos XML file and prepare its scene/component texts."""
    path = path.resolve()
    if path.suffix.lower() != ".xml":
        raise ValueError(f"Expected an XML file: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
        data = parse_caption(text)
        if not isinstance(data, dict):
            raise ValueError("Caption must be an object")
        narrative = data.get("comprehensive_t2i_caption")
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError("Missing or empty comprehensive_t2i_caption")
        scene = {k: v for k, v in data.items()
                 if k not in {"comprehensive_t2i_caption", "resolution", "aspect_ratio"}}
        validate_scene_draft(scene)
    except Exception as exc:
        raise ValueError(f"Invalid caption {path}: {exc}") from exc

    scene_id = str(uuid5(NAMESPACE_URL, os.path.normcase(str(path))))
    common = {"scene_id": scene_id, "filename": path.name, "source_path": str(path)}
    payload = {**common, "original_caption": data,
               "embedding_texts": {"scene": embedding_text(scene), "caption": embedding_text(narrative)}}
    components = []
    for key, value in scene.items():
        if key in {"subjects", "text_and_signage_elements"}:
            kind = "subject" if key == "subjects" else "signage"
            entries = [(kind, f"{key}[{i}]", item) for i, item in enumerate(value)]
        elif key == "quadrant_scan":
            entries = [("quadrant", f"quadrant_scan.{k}", v) for k, v in sorted(value.items())]
        else:
            entries = [("background" if key == "background_setting" else key, key, value)]
        for kind, field_path, item in entries:
            if has_content(item):
                components.append({"id": str(uuid5(UUID(scene_id), field_path)), "payload": {
                    **common, "component_type": kind, "field_path": field_path,
                    "original_value": item, "embedding_text": embedding_text(item)}})
    return {"id": scene_id, "payload": payload, "components": components}


async def prepare_collection(client: AsyncQdrantClient, collection: str,
                             dimension: int, vector_names: tuple[str, ...]) -> None:
    if await client.collection_exists(collection):
        vectors = (await client.get_collection(collection)).config.params.vectors
        if (not isinstance(vectors, dict) or set(vectors) != set(vector_names)
                or any(v.size != dimension or v.distance != models.Distance.COSINE
                       or v.multivector_config is not None for v in vectors.values())):
            raise ValueError(f"Incompatible vector configuration in {collection}")
        return
    await client.create_collection(collection_name=collection, vectors_config={
        name: models.VectorParams(size=dimension, distance=models.Distance.COSINE) for name in vector_names})
    for field in ("filename", "scene_id", "component_type", "field_path"):
        await client.create_payload_index(collection_name=collection, field_name=field,
                                         field_schema=models.PayloadSchemaType.KEYWORD, wait=True)


class VllmEncoder:
    """Keep all calls to a single synchronous vLLM engine on one worker thread."""

    def __init__(self, llm: Any, executor: ThreadPoolExecutor):
        self.llm = llm
        self.executor = executor

    async def embed(self, texts: list[str]) -> list[list[float]]:
        outputs = await asyncio.get_running_loop().run_in_executor(
            self.executor, partial(self.llm.embed, texts, use_tqdm=False))
        if len(outputs) != len(texts):
            raise ValueError("Local vLLM returned the wrong number of embeddings")
        return [output.outputs.embedding for output in outputs]


@asynccontextmanager
async def open_encoder(args: Args):
    if args.embedding_backend == "api":
        async with AsyncOpenAI(base_url=args.embedding_base_url,
                               api_key=os.getenv("EMBEDDING_API_KEY", "EMPTY"),
                               timeout=args.timeout) as encoder:
            yield encoder
        return

    # API mode must work without importing/installing vLLM or its GPU dependencies.
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ValueError(
            "Local vLLM requires the caption-vllm dependency group; install with "
            "'uv sync --extra caption-audit --group caption-vllm' on Linux/WSL") from exc
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="caption-vllm")
    try:
        llm = await asyncio.get_running_loop().run_in_executor(executor, partial(
            LLM, model=args.model, runner="pooling",
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            enforce_eager=args.vllm_enforce_eager))
        yield VllmEncoder(llm, executor)
    finally:
        # Await running inference even when another file worker fails/cancels.
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)


async def encode_texts(client: AsyncOpenAI | VllmEncoder, texts: list[str], args: Args) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        if isinstance(client, VllmEncoder):
            batch_vectors = await client.embed(batch)
        else:
            response = await client.embeddings.create(model=args.model, input=batch, encoding_format="float")
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise ValueError("Embedding API returned invalid or missing indexes")
            batch_vectors = [item.embedding for item in ordered]
        for vector in batch_vectors:
            if len(vector) != args.dimension or any(not math.isfinite(x) for x in vector):
                raise ValueError("Embedding backend returned wrong dimensions or nonfinite values")
            norm = math.sqrt(sum(x*x for x in vector))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("Embedding backend returned a zero or invalid vector")
            vectors.append([x/norm for x in vector])
    return vectors


def record_texts(record: dict) -> list[str]:
    texts = list(record["payload"]["embedding_texts"].values())
    texts.extend(c["payload"]["embedding_text"] for c in record["components"])
    return list(dict.fromkeys(texts))


def save_embeddings(folder: Path, record: dict, vectors: dict[str, list[float]], args: Args) -> None:
    """Atomically save one portable, uncompressed NPZ without Python pickle objects."""
    texts = record_texts(record)
    metadata = {"format_version": 1, "model": args.model, "dimension": args.dimension,
                "record": record, "texts": texts}
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{UUID(record['id'])}.npz"
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(stream, vectors=np.asarray([vectors[text] for text in texts], dtype="<f4"),
                     metadata=json.dumps(metadata, ensure_ascii=False, allow_nan=False))
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_embeddings(path: Path, args: Args) -> tuple[dict, dict[str, list[float]]]:
    """Load and validate a whole caption before any of its database writes."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].item())
            matrix = archive["vectors"]
        if metadata["format_version"] != 1:
            raise ValueError("Unsupported embedding format version")
        if metadata["model"] != args.model or metadata["dimension"] != args.dimension:
            raise ValueError("Embedding model/dimension differ from --model/--dimension")
        record, texts = metadata["record"], metadata["texts"]
        scene_id = str(UUID(record["id"]))
        if record["id"] != scene_id or record["payload"]["scene_id"] != scene_id:
            raise ValueError("Invalid scene ID")
        if set(record["payload"]["embedding_texts"]) != {"scene", "caption"}:
            raise ValueError("Expected scene and caption texts")
        component_ids = set()
        for component in record["components"]:
            component_id = str(UUID(component["id"]))
            if component["id"] != component_id or component_id in component_ids:
                raise ValueError("Invalid or duplicate component ID")
            component_ids.add(component_id)
            if component["payload"]["scene_id"] != scene_id:
                raise ValueError("Component belongs to a different scene")
        if (not isinstance(texts, list) or not all(isinstance(t, str) for t in texts)
                or texts != record_texts(record)):
            raise ValueError("Embedding texts do not match the record")
        if matrix.dtype.kind != "f" or matrix.dtype.itemsize != 4 or matrix.shape != (len(texts), args.dimension):
            raise ValueError("Invalid float32 embedding matrix or dimensions")
        if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix.astype(np.float64), axis=1) == 0):
            raise ValueError("Embedding matrix contains nonfinite or zero vectors")
        return record, dict(zip(texts, matrix.tolist(), strict=True))
    except Exception as exc:
        raise ValueError(f"Invalid embedding file {path}: {exc}") from exc


async def upload_record(qdrant: AsyncQdrantClient, record: dict,
                        vectors: dict[str, list[float]], args: Args) -> None:
    components = record["components"]
    # Construct all points before starting writes, including when importing an archive.
    points = [models.PointStruct(id=c["id"], payload=c["payload"],
              vector={"content": vectors[c["payload"]["embedding_text"]]}) for c in components]
    scene = models.PointStruct(id=record["id"], payload=record["payload"], vector={
        name: vectors[text] for name, text in record["payload"]["embedding_texts"].items()})
    for start in range(0, len(points), args.batch_size):
        await qdrant.upsert(args.components_collection, points=points[start:start + args.batch_size], wait=True)
    await qdrant.upsert(args.scenes_collection, points=[scene], wait=True)
    # Updating a file must not leave components removed from its JSON.
    obsolete = models.Filter(must=[models.FieldCondition(
        key="scene_id", match=models.MatchValue(value=record["id"]))])
    if components:
        obsolete.must_not = [models.HasIdCondition(has_id=[c["id"] for c in components])]
    await qdrant.delete(args.components_collection,
                        points_selector=models.FilterSelector(filter=obsolete), wait=True)


async def populate(encoder: AsyncOpenAI | VllmEncoder | None, qdrant: AsyncQdrantClient | None,
                   paths: list[Path], args: Args) -> int:
    remaining = iter(paths)
    uploaded = 0
    with tqdm(total=len(paths), desc="Files", unit="file") as progress:
        async def worker():
            nonlocal uploaded
            while (path := next(remaining, None)) is not None:
                if args.input_emb is not None:
                    record, vectors = await asyncio.to_thread(load_embeddings, path, args)
                else:
                    if encoder is None:
                        raise ValueError("An embedding encoder is required for XML input")
                    record = read_record(path)
                    unique = record_texts(record)
                    vectors = dict(zip(unique, await encode_texts(encoder, unique, args), strict=True))
                if args.output_emb is not None:
                    await asyncio.to_thread(save_embeddings, args.output_emb, record, vectors, args)
                else:
                    if qdrant is None:
                        raise ValueError("Qdrant is required unless --output-emb is supplied")
                    await upload_record(qdrant, record, vectors, args)
                uploaded += 1
                progress.update(1)
        async with asyncio.TaskGroup() as group:
            for _ in range(min(args.concurrency, len(paths))):
                group.create_task(worker())
    return uploaded


async def run(args: Args) -> int:
    paths = (collect_paths([args.input_emb], suffix=".npz") if args.input_emb is not None
             else collect_paths(args.input))
    if args.output_emb is not None:
        async with open_encoder(args) as encoder:
            return await populate(encoder, None, paths, args)

    async def import_to_qdrant(encoder):
        client = AsyncQdrantClient(url=args.qdrant_url, api_key=os.getenv("QDRANT_API_KEY"),
                                   timeout=math.ceil(args.timeout))
        try:
            await prepare_collection(client, args.scenes_collection, args.dimension, ("scene", "caption"))
            await prepare_collection(client, args.components_collection, args.dimension, ("content",))
            return await populate(encoder, client, paths, args)
        finally:
            await client.close()
    if args.input_emb is not None:
        return await import_to_qdrant(None)
    async with open_encoder(args) as encoder:
        return await import_to_qdrant(encoder)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    args = tyro.cli(Args)
    try:
        count = asyncio.run(run(args))
    except Exception as exc:
        if isinstance(exc, ExceptionGroup):
            exc = exc.exceptions[0]
        if isinstance(exc, (ValueError, FileNotFoundError)):
            log.error("%s", exc)
        else:
            log.error("Import failed (%s): %s; check the embedding backend and Qdrant logs",
                      type(exc).__name__, exc)
        return 1
    if args.output_emb is not None:
        log.info("Saved embeddings for %d files to %s", count, args.output_emb)
    else:
        log.info("Saved %d files to Qdrant", count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
