"""Store description JSONs and Qwen embeddings in the Qdrant collection description."""

import argparse
import json
import logging
import os
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from src.embeddings import (
    EMBEDDING_DIMENSION, MODEL_NAME, encode_documents, load_encoder,
)

log = logging.getLogger(__name__)
TEXT_FIELDS = ("foreground", "background")


@dataclass(frozen=True)
class Description:
    source: str
    point_id: int
    data: dict


def read_descriptions(input_dir):
    """Validate the selected corpus before any Qdrant calls or model downloads."""
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    files = sorted(p for p in input_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".json" and not p.name.startswith("_"))
    if not files:
        raise ValueError(f"No JSON found in {input_dir}")
    descriptions, errors = [], []
    for index, path in enumerate(files, start=1):
        source = path.relative_to(input_dir).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            descriptions.append(Description(source, index, data))
        except (OSError, ValueError) as exc:
            errors.append(f"{source}: {exc}")
    if errors:
        raise ValueError("Correggi i JSON prima del caricamento:\n" + "\n".join(errors[:20])
                         + (f"\n... {len(errors)} errori totali." if len(errors) > 20 else ""))
    return descriptions


def prepare_collection(client, collection, dimension=1024):
    from qdrant_client import models

    if client.collection_exists(collection):
        raise ValueError(f"Collection '{collection}' already exists")

    client.create_collection(
        collection_name=collection,
        vectors_config={
            field: models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE
            )
            for field in TEXT_FIELDS
        }
    )

    fields = {
        "lighting": "keyword",
        "weather": "keyword",
        "cars_count": "integer"
    }

    for field, field_type in fields.items():
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=field_type,
            wait=True
        )


def populate(client, descriptions, *, collection="description", batch_size=16,
             upload_batch_size=64):
    """Encode and upload every description to a newly created collection."""
    from qdrant_client import models

    prepare_collection(client, collection)
    model = None
    uploaded = 0
    with tqdm(total=len(descriptions), desc="Populating Qdrant", unit="record") as progress:
        for start in range(0, len(descriptions), upload_batch_size):
            batch = descriptions[start:start + upload_batch_size]
            if model is None:
                log.info("Loading %s", MODEL_NAME)
                model = load_encoder("cuda")
            # Deduplicate equal text within a batch without adding field labels or instructions.
            texts = list(dict.fromkeys(item.data[field] for item in batch for field in TEXT_FIELDS))
            encoded = encode_documents(model, texts, batch_size=batch_size)
            vectors = dict(zip(texts, encoded.tolist()))
            points = [models.PointStruct(
                id=item.point_id, payload=item.data,
                vector={field: vectors[item.data[field]] for field in TEXT_FIELDS},
            ) for item in batch]
            client.upsert(collection_name=collection, points=points, wait=True)
            uploaded += len(points)
            progress.set_postfix(uploaded=uploaded, refresh=False)
            progress.update(len(batch))
    return {"collection": collection, "model": MODEL_NAME, "dimension": EMBEDDING_DIMENSION,
            "records": len(descriptions), "uploaded": uploaded}


def cli_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default="description")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Encoder batch size (default: 16).")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size should be > 0.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        descriptions = read_descriptions(args.input)
        from qdrant_client import QdrantClient

        with closing(QdrantClient(url=args.url, api_key=os.getenv("QDRANT_API_KEY"), timeout=120)) as client:
            result = populate(client, descriptions, collection=args.collection,
                              batch_size=args.batch_size, upload_batch_size=args.batch_size)
    except Exception as exc:
        parser.exit(1, f"Loading aborted: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
