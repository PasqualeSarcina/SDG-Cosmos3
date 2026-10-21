import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"


def json_to_text(data: dict) -> str:
    """Converte la scena e la visibilità di ogni auto in testo per l'encoder."""
    if not isinstance(data, dict):
        raise ValueError("Il file deve contenere un singolo oggetto JSON.")

    parts = []

    for field in ("foreground", "background", "lighting", "weather"):
        value = data.get(field)
        if value is not None and str(value).strip():
            parts.append(f"{field}: {str(value).strip()}")

    cars_visibility = data.get("cars_visibility", [])
    if cars_visibility is None:
        cars_visibility = []
    if not isinstance(cars_visibility, list):
        raise ValueError("'cars_visibility' deve essere una lista.")

    cars_count = data.get("cars_count")
    if cars_count is None:
        cars_count = len(cars_visibility)

    parts.append(f"cars_count: {cars_count}")

    for index, visibility in enumerate(cars_visibility, start=1):
        if not isinstance(visibility, str):
            raise ValueError(f"La visibilità dell'auto {index} deve essere una stringa.")
        if visibility.strip():
            parts.append(f"car {index}: visibility: {visibility.strip()}")

    if not any(
        str(data.get(field) or "").strip()
        for field in ("foreground", "background", "lighting", "weather")
    ) and not any(visibility.strip() for visibility in cars_visibility):
        raise ValueError("Nessuna descrizione della scena trovata.")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Genera un embedding .npy per ciascun file JSON."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size deve essere maggiore di zero.")

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.is_dir():
        parser.error(f"Cartella di input inesistente: {input_dir}")

    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )

    if not files:
        print(f"Nessun file JSON trovato in {input_dir}")
        return

    # Controlla tutti gli input prima di creare gli embedding.
    texts = []
    errors = []

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            texts.append(json_to_text(data))
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")

    if errors:
        raise SystemExit(
            "Correggi questi file e riesegui lo script:\n"
            + "\n".join(errors)
        )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA non disponibile. Usa --device cpu.")

    print(f"File trovati: {len(files)}")
    print(f"Dispositivo: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    # Segnala descrizioni che superano il limite dell'encoder.
    too_long = []
    for path, text in zip(files, texts):
        tokens = model.tokenizer(
            text,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        if len(tokens) > model.max_seq_length:
            too_long.append(path.name)

    if too_long:
        print(
            f"ATTENZIONE: {len(too_long)} descrizioni superano "
            f"{model.max_seq_length} token e verranno troncate:"
        )
        for name in too_long:
            print(f"  - {name}")

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for path, embedding in zip(files, embeddings):
        output_path = output_dir / f"{path.stem}.npy"
        np.save(output_path, embedding.astype(np.float32))

    metadata = {
        "model": MODEL_NAME,
        "normalize_embeddings": False,
        "embedding_dimension": int(embeddings.shape[1]),
        "max_seq_length": int(model.max_seq_length),
        "files": [
            {
                "input": path.name,
                "embedding": f"{path.stem}.npy",
                "truncated": path.name in too_long,
            }
            for path in files
        ],
    }

    (output_dir / "_embedding_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Salvati {len(files)} embedding in: {output_dir}")


if __name__ == "__main__":
    main()
