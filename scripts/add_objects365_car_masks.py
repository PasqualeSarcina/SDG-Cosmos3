"""Add SAM3 Tracker instance masks to a Roboflow COCO Car dataset."""

import argparse
from collections import defaultdict
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from itertools import batched
import json
from math import ceil, floor, isfinite
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from tempfile import NamedTemporaryFile

from PIL import Image
from tqdm import tqdm


MODEL_ID = "facebook/sam3"
COCO_FILE = "_annotations.coco.json"
SPLITS = ("train", "valid")
JSON_OPTIONS = {"separators": (",", ":"), "allow_nan": False}


@dataclass
class Split:
    name: str
    coco: dict
    annotations: dict
    digest: str


@contextmanager
def atomic_output(path):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_split(source, name):
    raw = (source / name / COCO_FILE).read_bytes()
    coco = json.loads(raw)
    categories = coco["categories"]
    if len(categories) != 1 or categories[0]["name"].casefold() != "car":
        raise ValueError(f"{name}: expected a single Car category")

    image_ids = set()
    filenames = set()
    for image in coco["images"]:
        filename = image["file_name"]
        path = Path(filename)
        if path.name != filename or path.suffix.lower() not in {".jpg", ".jpeg"} or "\\" in filename:
            raise ValueError(f"{name}: expected a JPEG filename, got {filename!r}")
        filename_key = os.path.normcase(filename)
        if type(image["id"]) is not int or image["id"] in image_ids or filename_key in filenames:
            raise ValueError(f"{name}: invalid or duplicate image ID/filename: {image['id']} / {filename}")
        if any(type(image[key]) is not int or image[key] < 1 for key in ("width", "height")):
            raise ValueError(f"{name}: invalid dimensions for image ID {image['id']}")
        image_ids.add(image["id"])
        filenames.add(filename_key)

    annotations = defaultdict(list)
    annotation_ids = set()
    for row in coco["annotations"]:
        if type(row["id"]) is not int or row["id"] in annotation_ids:
            raise ValueError(f"{name}: invalid or duplicate annotation ID {row['id']}")
        if row["image_id"] not in image_ids or row["category_id"] != categories[0]["id"]:
            raise ValueError(f"{name}: annotation ID {row['id']} references an unknown image or category")
        annotations[row["image_id"]].append(row)
        annotation_ids.add(row["id"])
    return Split(name, coco, annotations, sha256(raw).hexdigest())


def prompt_box(bbox, width, height):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if any(type(value) not in (int, float) or not isfinite(value) for value in bbox):
        return None
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def open_progress(db, identity):
    # The exclusive lock survives commits and is released if Slurm kills the job.
    db.execute("PRAGMA locking_mode=EXCLUSIVE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("BEGIN IMMEDIATE")
    db.execute("CREATE TABLE IF NOT EXISTS run (identity TEXT NOT NULL)")
    db.execute("""
        CREATE TABLE IF NOT EXISTS completed (
            split TEXT, image_id INTEGER, source_stamp TEXT, masks TEXT,
            PRIMARY KEY (split, image_id)
        )
    """)
    encoded = json.dumps(identity, sort_keys=True)
    row = db.execute("SELECT identity FROM run").fetchone()
    if row is None:
        db.execute("INSERT INTO run VALUES (?)", (encoded,))
    elif row[0] != encoded:
        previous = json.loads(row[0])
        if previous.get("preview") is False:
            del previous["preview"]
        if previous == {**identity, "version": 1}:
            # Older checkpoints recorded failures before the crop retry was available.
            db.execute("""DELETE FROM completed WHERE instr(masks, '"discard_reason":') > 0""")
        elif previous != identity:
            raise ValueError("Output belongs to a different input or a sample run. Use a new --output directory.")
        db.execute("UPDATE run SET identity=?", (encoded,))
    db.commit()


def source_stamp(path):
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def read_jpeg(path, record, pixels):
    with Image.open(path) as image:
        expected = (record["width"], record["height"])
        if image.format != "JPEG" or image.size != expected:
            raise ValueError(f"{path}: expected JPEG {expected}, found {image.format} {image.size}")
        if pixels:
            return image.convert("RGB")


def materialize_jpeg(source, target, stamp):
    if target.is_file() and not target.is_symlink() and source_stamp(target) == stamp:
        return
    with atomic_output(target) as temporary:
        temporary.unlink()
        try:
            os.link(source.resolve(), temporary)
        except OSError:
            shutil.copy2(source, temporary)


class Sam3:
    def __init__(self, device):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self.torch = torch
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)
        self.processor = Sam3TrackerProcessor.from_pretrained(MODEL_ID)
        self.model = Sam3TrackerModel.from_pretrained(MODEL_ID).to(self.device).eval()

    def predict(self, images, boxes):
        inputs = self.processor(images=images, input_boxes=boxes, return_tensors="pt")
        sizes = inputs.pop("original_sizes")
        with self.torch.inference_mode():
            # The object axis keeps prompts separate while the encoder runs once per image.
            logits = self.model(**inputs.to(self.device), multimask_output=False).pred_masks.float().cpu()
        if logits.ndim != 5 or logits.shape[:3] != (len(images), len(boxes[0]), 1):
            raise RuntimeError(f"Unexpected SAM3 mask shape: {tuple(logits.shape)}")
        for masks, size in zip(logits, sizes, strict=True):
            finite = masks.isfinite().flatten(1).all(dim=1).tolist()
            binary = self.processor.post_process_masks(masks[None], [tuple(map(int, size))])[0]
            yield binary[:, 0].numpy(), finite


def encode_results(record, prompts, masks, finite, mask_utils, np, crop=None):
    width, height = record["width"], record["height"]
    left, top, right, bottom = crop or (0, 0, width, height)
    expected = (len(prompts), bottom - top, right - left)
    if masks.shape != expected or masks.dtype != np.bool_:
        raise RuntimeError(f"Image ID {record['id']}: expected binary masks of shape {expected}, got {masks.shape}")
    result = {}
    for (annotation_id, box), mask, valid in zip(prompts, masks, finite, strict=True):
        x1, y1, x2, y2 = box
        if not valid or not mask.any():
            outcome = {"discard_reason": "empty mask" if valid else "non-finite SAM3 output"}
        elif not mask[floor(y1):ceil(y2), floor(x1):ceil(x2)].any():
            outcome = {"discard_reason": "mask does not intersect its prompt box"}
        else:
            if crop is not None:
                mask = np.pad(mask, ((top, height - bottom), (left, width - right)))
            rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
            rle["counts"] = rle["counts"].decode("ascii")
            outcome = {"segmentation": rle, "area": int(mask.sum())}
        result[str(annotation_id)] = outcome
    return result


def retry_failed_masks(image, record, prompts, results, predict, batch_size, mask_utils, np):
    retries = []
    for annotation_id, (x1, y1, x2, y2) in prompts:
        if "discard_reason" not in results[str(annotation_id)]:
            continue
        margin = max(16, max(x2 - x1, y2 - y1) / 2)
        left, top = max(0, floor(x1 - margin)), max(0, floor(y1 - margin))
        right, bottom = min(image.width, ceil(x2 + margin)), min(image.height, ceil(y2 + margin))
        crop = (left, top, right, bottom)
        box = [x1 - left, y1 - top, x2 - left, y2 - top]
        retries.append((annotation_id, box, crop))

    for batch in batched(retries, batch_size):
        images = [image.crop(crop) for _, _, crop in batch]
        boxes = [[box] for _, box, _ in batch]
        for (annotation_id, box, crop), (masks, finite) in zip(batch, predict(images, boxes), strict=True):
            key = str(annotation_id)
            result = encode_results(record, [(annotation_id, box)], masks, finite, mask_utils, np, crop)[key]
            if "discard_reason" in result:
                results[key]["discard_reason"] += f"; crop retry: {result['discard_reason']}"
            else:
                results[key] = {**result, "recovered_on_crop": True}


def process_split(args, split, predict, mask_utils, np, stamps, notify):
    records = split.coco["images"]
    groups = defaultdict(list)
    for record in records:
        groups[len(split.annotations.get(record["id"], ()))].append(record)

    failures = 0

    def report(message):
        nonlocal failures
        failures += 1
        notify(("error", f"{split.name} [{args.device}]: {message}"))

    def save(record, source, stamp, results):
        if source_stamp(source) != stamp:
            raise OSError(f"{source}: JPEG changed during processing")
        notify(("result", (split.name, record["id"], stamp, json.dumps(results, **JSON_OPTIONS))))

    for group in groups.values():
        for batch in batched(group, args.batch_size):
            pending_by_prompt_count = defaultdict(list)
            reused = skipped = 0
            for record in batch:
                image_id = record["id"]
                rows = split.annotations.get(image_id, ())
                prompts = []
                discarded = {}
                for row in rows:
                    box = prompt_box(row.get("bbox"), record["width"], record["height"])
                    if box is None:
                        discarded[str(row["id"])] = {"discard_reason": f"unusable box {row.get('bbox')!r}"}
                    else:
                        prompts.append((row["id"], box))

                source = args.input / split.name / record["file_name"]
                try:
                    stamp = source_stamp(source)
                    cached = stamps.get(image_id) == stamp
                    image = read_jpeg(source, record, pixels=not cached)
                except (OSError, ValueError) as exc:
                    report(f"image ID {image_id}: {source}: {exc}")
                    skipped += 1
                    continue
                if cached:
                    reused += 1
                elif prompts:
                    # After discards, SAM3 still needs the same prompt count per image in a batch.
                    pending_by_prompt_count[len(prompts)].append((record, source, stamp, prompts, image, discarded))
                else:
                    save(record, source, stamp, discarded)

            if reused or skipped:
                notify(("checked", (reused, skipped)))
            for pending in pending_by_prompt_count.values():
                images = [job[4] for job in pending]
                boxes = [[box for _, box in job[3]] for job in pending]
                for (record, source, stamp, prompts, image, discarded), (masks, finite) in zip(
                    pending, predict(images, boxes), strict=True,
                ):
                    results = encode_results(record, prompts, masks, finite, mask_utils, np)
                    retry_failed_masks(image, record, prompts, results, predict, args.batch_size, mask_utils, np)
                    save(record, source, stamp, {**discarded, **results})
    return failures


def infer_splits(args, splits, stamps, notify):
    import numpy as np
    from pycocotools import mask as mask_utils

    predictor = None

    def predict(images, boxes):
        nonlocal predictor
        if predictor is None:
            predictor = Sam3(args.device)
        return predictor.predict(images, boxes)

    return sum(
        process_split(args, split, predict, mask_utils, np, stamps[split.name], notify)
        for split in splits
    )


def shard_splits(splits, gpu_count):
    shards = [[] for _ in range(gpu_count)]
    for split in splits:
        # Spread images with the same number of prompts across the GPUs.
        records = sorted(split.coco["images"], key=lambda row: len(split.annotations.get(row["id"], ())))
        for rank, shard in enumerate(shards):
            images = records[rank::gpu_count]
            annotations = {row["id"]: split.annotations[row["id"]] for row in images if row["id"] in split.annotations}
            shard.append(Split(split.name, {"images": images}, annotations, split.digest))
    return shards


def infer_worker(args, splits, stamps, pipe):
    with pipe:
        failures = infer_splits(args, splits, stamps, pipe.send)
        pipe.send(("done", failures))


def infer_on_gpus(args, splits, stamps, notify):
    context = mp.get_context("spawn")
    workers = {}
    failures = 0
    try:
        for rank, shard in enumerate(shard_splits(splits, args.gpus)):
            worker_args = argparse.Namespace(input=args.input, batch_size=args.batch_size, device=f"cuda:{rank}")
            shard_stamps = {
                split.name: {
                    row["id"]: stamps[split.name][row["id"]]
                    for row in split.coco["images"] if row["id"] in stamps[split.name]
                }
                for split in shard
            }
            reader, writer = context.Pipe(duplex=False)
            process = context.Process(
                name=f"SAM3 cuda:{rank}", target=infer_worker,
                args=(worker_args, shard, shard_stamps, writer),
            )
            try:
                with writer:
                    process.start()
            except BaseException:
                reader.close()
                raise
            workers[reader] = process

        active = set(workers)
        while active:
            for reader in wait(active):
                process = workers[reader]
                try:
                    event = reader.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f"{process.name} stopped before completing its shard. "
                                       "See the worker log and resume with the same output.") from exc
                if event[0] == "done":
                    failures += event[1]
                    process.join()
                    if process.exitcode != 0:
                        raise RuntimeError(f"{process.name} exited with status {process.exitcode}")
                    active.remove(reader)
                else:
                    notify(event)
    finally:
        for process in workers.values():
            if process.is_alive():
                process.terminate()
        for reader, process in workers.items():
            reader.close()
            process.join()
            process.close()
    return failures


def write_split(db, args, split, target, mask_utils, stale_jpegs, progress):
    records = split.coco["images"]
    kept = []
    count = discarded = recovered = 0
    with target.open("w", encoding="utf-8") as stream:
        stream.write("{")
        for key, value in split.coco.items():
            if key in {"images", "annotations"}:
                continue
            stream.write(f"{json.dumps(key)}:")
            json.dump(value, stream, **JSON_OPTIONS)
            stream.write(",")
        stream.write('"annotations":[')
        for record in records:
            image_id = record["id"]
            source = args.input / split.name / record["file_name"]
            jpeg = args.output / split.name / record["file_name"]
            stamp = source_stamp(source)
            saved = db.execute(
                "SELECT masks FROM completed WHERE split=? AND image_id=? AND source_stamp=?",
                (split.name, image_id, stamp),
            ).fetchone()
            if saved is None:
                raise ValueError(f"{split.name}: image ID {image_id} has no complete checkpoint")
            results = json.loads(saved[0])
            rows = split.annotations.get(image_id, ())
            if results.keys() != {str(row["id"]) for row in rows}:
                raise ValueError(f"{split.name}: image ID {image_id} has incomplete annotation IDs")
            annotations = []
            for row in rows:
                result = results[str(row["id"])]
                if "discard_reason" in result:
                    tqdm.write(f"{split.name}: discarded annotation ID {row['id']} (image ID {image_id}): "
                               f"{result['discard_reason']}", file=sys.stderr)
                    discarded += 1
                    continue
                rle, area = result["segmentation"], result["area"]
                if (rle["size"] != [record["height"], record["width"]] or not isinstance(rle["counts"], str)
                        or not 0 < area <= record["height"] * record["width"] or mask_utils.area(rle) != area):
                    raise ValueError(f"{split.name}: invalid RLE or area for annotation ID {row['id']}")
                recovered += result.get("recovered_on_crop", False)
                annotations.append({**row, "segmentation": rle, "area": area})

            missing = len(rows) - len(annotations)
            excluded = missing > 0 if split.name == "valid" else bool(rows and not annotations)
            if excluded:
                tqdm.write(f"{split.name}: excluded image ID {image_id}: {missing}/{len(rows)} missing masks",
                           file=sys.stderr)
                if jpeg.is_file() or jpeg.is_symlink():
                    stale_jpegs.append(jpeg)
                progress.update()
                continue

            materialize_jpeg(source, jpeg, stamp)
            read_jpeg(jpeg, record, pixels=False)
            kept.append(record)
            for annotation in annotations:
                if count:
                    stream.write(",")
                json.dump(annotation, stream, **JSON_OPTIONS)
                count += 1
            progress.update()
        stream.write('],"images":')
        json.dump(kept, stream, **JSON_OPTIONS)
        stream.write("}\n")
    return (f"{split.name}: {len(kept)} JPEGs and {count} masks exported; "
            f"{len(records) - len(kept)} images excluded, {discarded} annotations discarded, "
            f"{recovered} masks recovered on crops")


def run(args):
    from pycocotools import mask as mask_utils

    if args.gpus is not None:
        import torch

        visible = torch.cuda.device_count()
        if args.gpus > visible:
            raise ValueError(f"Requested {args.gpus} GPUs, but only {visible} CUDA devices are visible.")
        args.device = "cuda:0"

    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.input.is_relative_to(args.output) or args.output.is_relative_to(args.input):
        raise ValueError("--input and --output must be separate directory trees")
    splits = [read_split(args.input, name) for name in SPLITS]
    progress_path = args.output / ".sam3-progress.sqlite3"
    if progress_path.is_symlink():
        raise ValueError(f"Refusing symlink at {progress_path}")
    for name in SPLITS:
        directory = args.output / name
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError(f"Expected a regular directory at {directory}")
        if not progress_path.exists() and directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Refusing to overwrite existing dataset at {directory}")
        directory.mkdir(parents=True, exist_ok=True)

    identity = {"version": 2, "model": MODEL_ID, "input": str(args.input),
                "annotations": {split.name: split.digest for split in splits}}
    interactive = sys.stderr.isatty()
    bar_options = {
        "total": sum(len(split.coco["images"]) for split in splits),
        "unit": "img", "dynamic_ncols": interactive, "mininterval": 1 if interactive else 15,
    }
    with closing(sqlite3.connect(progress_path, timeout=1)) as db:
        open_progress(db, identity)
        stamps = {
            split.name: dict(db.execute("SELECT image_id, source_stamp FROM completed WHERE split=?", (split.name,)))
            for split in splits
        }
        infer = infer_on_gpus if args.gpus and args.gpus > 1 else infer_splits
        description = f"SAM3 ({args.gpus} GPU)" if args.gpus else "SAM3"
        with tqdm(desc=description, colour="cyan" if interactive else None, **bar_options) as progress:
            reused = skipped = 0

            # Only the coordinator updates the bar and commits results.
            def notify(event):
                nonlocal reused, skipped
                kind, data = event
                if kind == "error":
                    tqdm.write(data, file=sys.stderr)
                elif kind == "checked":
                    cached, invalid = data
                    reused += cached
                    skipped += invalid
                    progress.set_postfix(cached=reused, invalid=skipped, refresh=False)
                    progress.update(cached + invalid)
                else:
                    with db:
                        db.execute("INSERT OR REPLACE INTO completed VALUES (?, ?, ?, ?)", data)
                    progress.update()

            failures = infer(args, splits, stamps, notify)
        if failures:
            raise ValueError(f"{failures} validation failures. See image/annotation IDs above. Coverage is incomplete.")

        # Prepare both JSON files before publishing either of them.
        stale_jpegs = []
        summaries = []
        colour = "green" if interactive else None
        with tqdm(desc="COCO export", colour=colour, **bar_options) as progress, ExitStack() as stack:
            for split in splits:
                progress.set_postfix_str(split.name, refresh=False)
                target = stack.enter_context(atomic_output(args.output / split.name / COCO_FILE))
                summaries.append(write_split(db, args, split, target, mask_utils, stale_jpegs, progress))
        for jpeg in stale_jpegs:
            jpeg.unlink(missing_ok=True)
        for summary in summaries:
            print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Requires the sam dependency group and access to facebook/sam3 on Hugging Face. "
        "Unusable boxes are discarded. Failed masks are retried on a crop; remaining failures discard the annotation. "
        "train images that lose all annotations and valid images with missing masks are excluded. "
        "Re-run with the same input and output to resume. Use a fresh output if the source JSON changes.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Source directory containing train/ and valid/")
    parser.add_argument("--output", required=True, type=Path, help="Separate output directory")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", default="auto", help="PyTorch device (default: CUDA if available, otherwise CPU)")
    devices.add_argument(
        "--gpus", type=int, help="Number of visible CUDA GPUs on this node (one process per GPU)",
    )
    parser.add_argument("--batch-size", default=32, type=int, help="Maximum images per GPU forward pass (default: 32)")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.gpus is not None and args.gpus < 1:
        parser.error("--gpus must be positive")
    try:
        run(args)
    except (ImportError, KeyError, OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved COCO instance masks to {args.output}")


if __name__ == "__main__":
    main()
