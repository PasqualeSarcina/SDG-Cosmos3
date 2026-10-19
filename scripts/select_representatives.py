"""Select real scene representatives using DADO's intra-cluster similarity.

Each NAME.npy must contain one embedding; NAME is used as the scene ID.
No LLM calls, model loading, or embedding normalization are performed.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans


def representative_scores(vectors, block_size=256):
    """Exact mean 1/(1 + squared Euclidean distance), including self similarity.

    DADO equations (6)-(7). Uses O(block_size**2) temporary distances
    instead of retaining the full pairwise matrix. Time remains quadratic
    in the number of members of this cluster.
    """
    count = len(vectors)
    if count == 0:
        raise ValueError("Cannot score an empty cluster.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    scores = np.empty(count, dtype=np.float64)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        totals = np.zeros(stop - start, dtype=np.float64)
        for other in range(0, count, block_size):
            distances = cdist(
                vectors[start:stop],
                vectors[other:other + block_size],
                metric="sqeuclidean",
            )
            distances += 1.0
            np.reciprocal(distances, out=distances)
            totals += distances.sum(axis=1)
        scores[start:stop] = totals / count
    return scores


def load_inputs(embeddings_dir):
    paths = sorted(embeddings_dir.glob("*.npy"))
    if not paths:
        raise ValueError(f"No .npy files found in {embeddings_dir}")

    records, vectors = [], []
    dimension = None
    for path in paths:
        vector = np.load(path, allow_pickle=False)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"{path.name}: expected one nonempty 1D embedding.")
        if vector.dtype.kind != "f":
            raise ValueError(f"{path.name}: expected floating-point numbers.")
        if not np.isfinite(vector).all():
            raise ValueError(f"{path.name}: contains NaN or infinite values.")
        if dimension is None:
            dimension = vector.size
        if vector.size != dimension:
            raise ValueError(f"{path.name}: inconsistent embedding dimension.")

        records.append({
            "id": path.stem,
            "embedding_file": path.name,
        })
        vectors.append(vector.astype(np.float32))

    # Copies metadata from the embedding script, if available.
    metadata_path = embeddings_dir / "_embedding_metadata.json"
    metadata = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        expected_files = {
            entry["embedding"] for entry in metadata.get("files", [])
            if "embedding" in entry
        }
        if expected_files and expected_files != {path.name for path in paths}:
            raise ValueError(
                "Embedding files differ from _embedding_metadata.json. "
                "Use a clean folder containing a single complete embedding run."
            )

    return records, np.stack(vectors), metadata


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=100)
    parser.add_argument("--kmeans-batch-size", type=int, default=1024)
    parser.add_argument("--distance-block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    for name in ("clusters", "kmeans_batch_size", "distance_block_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")

    embeddings_dir = args.embeddings.resolve()
    output_dir = args.output.resolve()
    if not embeddings_dir.is_dir():
        parser.error("Embedding directory must exist.")
    if output_dir == embeddings_dir:
        parser.error("Use a separate output directory.")

    records, matrix, embedding_metadata = load_inputs(embeddings_dir)
    if args.clusters > len(records):
        parser.error("--clusters cannot exceed the number of embeddings.")

    print(f"Loaded {len(records)} embeddings of dimension {matrix.shape[1]}.", flush=True)
    print(f"Clustering into {args.clusters} groups...", flush=True)
    kmeans = MiniBatchKMeans(
        n_clusters=args.clusters,
        random_state=args.seed,
        batch_size=args.kmeans_batch_size,
        n_init=10,
    )
    kmeans.fit(matrix)
    labels = kmeans.predict(matrix)

    representatives, clusters = [], []
    occupied = np.unique(labels)
    if len(occupied) != args.clusters:
        warnings.warn(
            f"Only {len(occupied)} clusters have members; "
            "empty clusters are omitted. Consider fewer clusters."
        )

    for index, cluster_id in enumerate(occupied, start=1):
        members = np.flatnonzero(labels == cluster_id)
        scores = representative_scores(matrix[members], args.distance_block_size)
        winner = int(members[int(np.argmax(scores))])
        record = records[winner]
        representatives.append({
            "cluster_id": int(cluster_id),
            "cluster_size": len(members),
            "average_intra_cluster_similarity": float(scores.max()),
            **record,
        })
        clusters.append({
            "cluster_id": int(cluster_id),
            "size": len(members),
            "representative_id": record["id"],
            "member_ids": [records[int(i)]["id"] for i in members],
        })
        print(
            f"[{index}/{len(occupied)}] cluster {cluster_id}: "
            f"{len(members)} scenes; representative={record['id']}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "representatives.json", representatives)
    write_json(output_dir / "clusters.json", clusters)
    print(f"Saved {len(representatives)} representatives in {output_dir}")


if __name__ == "__main__":
    main()
