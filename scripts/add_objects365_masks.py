"""Add SAM3 masks to every annotated class in Roboflow COCO train/ and valid/, with bbox masks for crowds."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager, nullcontext
from hashlib import file_digest
from importlib.util import find_spec
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ATTENTION_BACKENDS = ("sdpa", "eager")


def dumps(value):
    return json.dumps(value, **JSON_OPTIONS)


@contextmanager
def atomic_output(path):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_stamp(path):
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def coco_entries(path):
    """Stream one image/annotation at a time, even when COCO keys are reordered."""
    import ijson

    def value(events, first):
        _, event, item = first
        if event not in {"start_map", "start_array"}:
            if event in {"string", "number", "boolean", "null"}:
                return item
            raise ValueError(f"{path}: malformed COCO value")
        builder = ijson.ObjectBuilder()
        builder.event(event, item)
        depth = 1
        for _, event, item in events:
            builder.event(event, item)
            depth += (event in {"start_map", "start_array"}) - (event in {"end_map", "end_array"})
            if depth == 0:
                return builder.value
        raise ValueError(f"{path}: truncated COCO value")

    with path.open("rb") as stream:
        events = iter(ijson.parse(stream, use_float=True))
        if next(events, None) != ("", "start_map", None):
            raise ValueError(f"{path}: expected a COCO object")
        seen = set()
        for prefix, event, key in events:
            if prefix == "" and event == "end_map":
                break
            if prefix != "" or event != "map_key" or key in seen:
                raise ValueError(f"{path}: invalid or duplicate COCO key {key!r}")
            seen.add(key)
            first = next(events)
            if key in {"images", "annotations"}:
                if first != (key, "start_array", None):
                    raise ValueError(f"{path}: {key} must be an array")
                for item in events:
                    if item == (key, "end_array", None):
                        break
                    yield key, value(events, item)
            else:
                yield key, value(events, first)
        if not {"images", "annotations", "categories"} <= seen:
            raise ValueError(f"{path}: missing images, annotations or categories")
        if next(events, None) is not None:
            raise ValueError(f"{path}: data after the COCO object")


def require_id(row, key, context):
    if type(row.get(key)) is not int:
        raise ValueError(f"{context}: {key} must be an integer")
    return row[key]


def validate_image(row, split):
    require_id(row, "id", split)
    filename = row["file_name"]
    path = Path(filename)
    if path.name != filename or "\\" in filename or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"{split}: expected a flat Roboflow image filename, got {filename!r}")
    if any(type(row.get(key)) is not int or row[key] < 1 for key in ("width", "height")):
        raise ValueError(f"{split}: invalid dimensions for image ID {row['id']}")


def prepare_index(args, digests):
    """Keep the complete dataset on disk instead of copying millions of rows to each GPU."""
    path = args.output / ".sam3-source.sqlite3"
    if path.is_symlink():
        raise ValueError(f"Refusing symlink at {path}")
    if path.exists():
        with closing(sqlite3.connect(path)) as db:
            saved = dict(db.execute("SELECT name, digest FROM splits"))
        if saved == digests:
            return path
        raise ValueError("Source annotations changed. Use a new --output directory.")

    with atomic_output(path) as temporary, closing(sqlite3.connect(temporary)) as db:
        db.executescript("""
            CREATE TABLE splits (name TEXT PRIMARY KEY, digest TEXT, metadata TEXT, image_count INTEGER);
            CREATE TABLE images (
                split TEXT, image_id INTEGER, ordinal INTEGER, filename TEXT, payload TEXT,
                PRIMARY KEY (split, image_id), UNIQUE (split, filename)
            );
            CREATE TABLE annotations (
                split TEXT, annotation_id INTEGER, image_id INTEGER, category_id INTEGER, payload TEXT,
                PRIMARY KEY (split, annotation_id)
            );
        """)
        expected_categories = None
        for split in SPLITS:
            metadata = {}
            count = 0
            print(f"{split}: indexing all classes and bounding boxes", flush=True)
            with db:
                for key, row in coco_entries(args.input / split / COCO_FILE):
                    if key == "images":
                        validate_image(row, split)
                        db.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?)",
                                   (split, row["id"], count, os.path.normcase(row["file_name"]), dumps(row)))
                        count += 1
                    elif key == "annotations":
                        ids = [require_id(row, key, split) for key in ("id", "image_id", "category_id")]
                        db.execute("INSERT INTO annotations VALUES (?, ?, ?, ?, ?)", (split, *ids, dumps(row)))
                    else:
                        metadata[key] = row
                categories = metadata["categories"]
                names = {}
                for category in categories:
                    category_id = require_id(category, "id", split)
                    name = category.get("name")
                    if category_id in names or not isinstance(name, str) or not name.strip():
                        raise ValueError(f"{split}: invalid or duplicate category ID/name {category!r}")
                    names[category_id] = name
                if not names:
                    raise ValueError(f"{split}: no categories")
                if expected_categories is not None and names != expected_categories:
                    raise ValueError("Train and validation category IDs/names do not match")
                expected_categories = names
                unknown = {row[0] for row in db.execute(
                    "SELECT DISTINCT category_id FROM annotations WHERE split=?", (split,),
                )} - names.keys()
                if unknown:
                    raise ValueError(f"{split}: unknown category IDs {sorted(unknown)}")
                db.execute("INSERT INTO splits VALUES (?, ?, ?, ?)",
                           (split, digests[split], dumps(metadata), count))
        db.executescript("""
            CREATE INDEX annotation_images ON annotations (split, image_id);
            CREATE INDEX image_order ON images (split, ordinal);
        """)
        orphan = db.execute("""
            SELECT a.split, a.annotation_id FROM annotations a LEFT JOIN images i
            ON a.split=i.split AND a.image_id=i.image_id WHERE i.image_id IS NULL LIMIT 1
        """).fetchone()
        if orphan:
            raise ValueError(f"{orphan[0]}: annotation ID {orphan[1]} references an unknown image")
        db.commit()
    return path


def prompt_box(bbox, width, height):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if any(type(value) not in (int, float) or not isfinite(value) for value in bbox):
        return None
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def read_image(path, record, pixels=True):
    with Image.open(path) as image:
        expected = (record["width"], record["height"])
        if image.size != expected:
            raise ValueError(f"{path}: expected dimensions {expected}, found {image.size}")
        if pixels:
            return image.convert("RGB")
    return None


def materialize_image(source, target, stamp):
    if target.is_file() and not target.is_symlink() and source_stamp(target) == stamp:
        return
    with atomic_output(target) as temporary:
        temporary.unlink()
        try:
            os.link(source.resolve(), temporary)
        except OSError:
            shutil.copy2(source, temporary)


def select_mask(logits, scores, box, size, torch):
    """Associate one candidate to the prompt bbox, without a geometric rejection threshold.

    No SAM detection box or predicted class is exported. Work at decoder resolution
    and resize only the selected mask, rather than all 200 candidates.
    """
    if logits.ndim != 3 or scores.shape != logits.shape[:1]:
        raise RuntimeError(f"Unexpected SAM3 mask/score shapes: {logits.shape}, {scores.shape}")
    finite = logits.isfinite().flatten(1).all(1) & scores.isfinite()
    binary = logits > 0
    height, width = logits.shape[-2:]
    columns, rows = binary.any(1), binary.any(2)
    xs = torch.arange(width, device=logits.device)[None]
    ys = torch.arange(height, device=logits.device)[None]
    x1 = torch.where(columns, xs, width).amin(1).float() / width
    y1 = torch.where(rows, ys, height).amin(1).float() / height
    x2 = (torch.where(columns, xs, -1).amax(1).float() + 1) / width
    y2 = (torch.where(rows, ys, -1).amax(1).float() + 1) / height
    image_height, image_width = size
    bx1, by1, bx2, by2 = [v / scale for v, scale in zip(box, (image_width, image_height) * 2, strict=True)]
    intersection = ((x2.clamp(max=bx2) - x1.clamp(min=bx1)).clamp(min=0)
                    * (y2.clamp(max=by2) - y1.clamp(min=by1)).clamp(min=0))
    union = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) + (bx2 - bx1) * (by2 - by1) - intersection
    iou = intersection / union.clamp(min=1e-12)
    valid = finite & columns.any(1)
    # Stable sorting makes IoU primary and the SAM score secondary.
    by_score = scores.argsort(descending=True, stable=True)
    ranked = by_score[iou[by_score].masked_fill(~valid[by_score], -1).argsort(descending=True, stable=True)]
    mask = torch.nn.functional.interpolate(
        logits[ranked[:1], None].float().sigmoid(), size=size, mode="bilinear", align_corners=False,
    )[0, 0] > 0.5
    mask &= valid[ranked[0]]
    # A single device-to-host synchronization; the empty-mask check then runs on CPU.
    mask = mask.cpu().numpy()
    if not mask.any():
        return None
    return mask


class OptimizedCall:
    """Compile a tensor-only stage, falling back once if compilation/capture fails."""

    def __init__(self, torch, function, enabled, mode, name):
        self.torch, self.function, self.name = torch, function, name
        self.compiled = None
        if enabled:
            try:
                self.compiled = torch.compile(function, mode=mode, fullgraph=False, dynamic=False)
            except Exception as exc:
                self.fallback(exc)

    def fallback(self, exc):
        # Dynamo/Inductor can wrap a CUDA OOM in a compiler exception.
        pending, seen = [exc], set()
        while pending:
            error = pending.pop()
            if id(error) in seen:
                continue
            seen.add(id(error))
            if isinstance(error, self.torch.OutOfMemoryError):
                raise exc
            pending.extend(cause for cause in (getattr(error, "inner_exception", None),
                                               error.__cause__, error.__context__)
                           if isinstance(cause, BaseException))
        self.compiled = None
        tqdm.write(f"SAM3 {self.name}: compile disabled; using eager execution: {exc}", file=sys.stderr)

    def __call__(self, *args, **kwargs):
        if self.compiled is not None:
            try:
                # Encoder features are cloned before a decoder step can reuse graph memory.
                self.torch.compiler.cudagraph_mark_step_begin()
                return self.compiled(*args, **kwargs)
            except Exception as exc:
                self.fallback(exc)
        return self.function(*args, **kwargs)


def set_config_dtype(config, dtype):
    """Propagate precision to nested backbones before loading the model."""
    config.dtype = dtype
    for name in config.sub_configs:
        child = getattr(config, name, None)
        if child is not None:
            set_config_dtype(child, dtype)


def worker_cpu_threads(gpus):
    """Share the job's CPU allocation across GPU workers, including their compilers."""
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(4, cpu_count // (gpus or 1)))


class Sam3:
    def __init__(self, args):
        import torch
        from transformers import AutoConfig, Sam3Model, Sam3Processor

        self.torch = torch
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                                   if args.device == "auto" else args.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        dtype = args.dtype
        if dtype == "auto":
            dtype = ("bfloat16" if self.device.type == "cuda"
                     and torch.cuda.is_bf16_supported(including_emulation=False) else "float32")
        self.dtype = getattr(torch, dtype)
        if self.device.type == "cpu" and self.dtype != torch.float32:
            raise ValueError("SAM3 bbox prompts require --dtype float32 on CPU (torchvision ROI Align limitation)")
        self.batch_size, self.image_batch_size = args.batch_size, args.image_batch_size
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
            torch.backends.cudnn.allow_tf32 = args.tf32
            torch.backends.cudnn.benchmark = True
        self.attn_implementation = args.attn_implementation
        if self.attn_implementation not in ATTENTION_BACKENDS:
            raise ValueError(f"--attn-implementation must be one of {ATTENTION_BACKENDS}")
        available = self.device.type == "cuda" and sys.platform.startswith("linux") and find_spec("triton") is not None
        if args.compile == "on" and not available:
            raise ValueError("--compile on requires Linux, CUDA and Triton; use --compile off on this platform")
        enabled = args.compile != "off" and available
        self.processor = Sam3Processor.from_pretrained(MODEL_ID)
        config = AutoConfig.from_pretrained(MODEL_ID)
        set_config_dtype(config, self.dtype)
        self.model = Sam3Model.from_pretrained(
            MODEL_ID, config=config, dtype=self.dtype, attn_implementation=self.attn_implementation,
        )
        self.model = self.model.to(self.device).eval()
        self.configure_runtime(enabled, args.compile_mode, getattr(args, "compile_scope", "all"),
                               getattr(args, "compile_recompile_limit", 128), getattr(args, "gpus", None))
        compile_description = (f"{args.compile_mode}, scope={self.compile_scope}, variants={self.compile_recompile_limit}, "
                               f"compiler threads={self.compile_threads}"
                               if enabled else "off")
        print(f"SAM3 [{self.device}]: dtype={dtype}, attention={self.attn_implementation}, "
              f"compile={compile_description}, text cache=on", flush=True)

    def configure_runtime(self, compile_enabled=False, compile_mode="reduce-overhead", compile_scope="all",
                          recompile_limit=128, gpus=None):
        self.compile_scope, self.compile_recompile_limit = compile_scope, recompile_limit
        self.compile_threads = None
        if compile_enabled:
            from torch._dynamo import config as dynamo_config
            from torch._inductor import config as inductor_config

            # Static batch buckets, configuration types and AMP dtypes need separate variants.
            # Surface limit hits so the wrapper can log and disable the affected stage.
            dynamo_config.recompile_limit = recompile_limit
            dynamo_config.accumulated_recompile_limit = max(dynamo_config.accumulated_recompile_limit,
                                                           16 * recompile_limit)
            dynamo_config.fail_on_recompile_limit_hit = True
            dynamo_config.suppress_errors = False
            # Each spawned GPU process owns an Inductor pool. Avoid N x 32 compiler
            # subprocesses competing for the same node's CPU allocation and memory.
            self.compile_threads = int(os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", worker_cpu_threads(gpus)))
            if self.compile_threads < 1:
                raise ValueError("TORCHINDUCTOR_COMPILE_THREADS must be positive")
            inductor_config.compile_threads = self.compile_threads
        self.text_cache = {}
        self.encode_images = OptimizedCall(
            self.torch, self.model.get_vision_features, compile_enabled, compile_mode, "image encoder",
        )

        def decode(**inputs):
            outputs = self.model(**inputs)
            return outputs.pred_masks, outputs.pred_logits

        self.decode_prompts = OptimizedCall(self.torch, decode, compile_enabled and compile_scope == "all",
                                            compile_mode, "prompt decoder")

    def text_features(self, texts):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        torch = self.torch
        missing = list(dict.fromkeys(text for text in texts if text not in self.text_cache))
        if missing:
            inputs = self.processor(text=missing, return_tensors="pt").to(self.device)
            features = self.model.get_text_features(**inputs, return_dict=True).pooler_output
            for index, text in enumerate(missing):
                self.text_cache[text] = (features[index].detach().clone(), inputs["attention_mask"][index].clone())
        features = torch.stack([self.text_cache[text][0] for text in texts])
        attention_mask = torch.stack([self.text_cache[text][1] for text in texts])
        return BaseModelOutputWithPooling(pooler_output=features), attention_mask

    def predict(self, images, prompts):
        """Yield (image index, annotation ID, mask), with one positive bbox + text per request."""
        torch = self.torch
        autocast = (torch.autocast(self.device.type, dtype=self.dtype)
                    if self.dtype != torch.float32 else nullcontext())
        with torch.inference_mode(), autocast:
            encoder_images = images
            if self.encode_images.compiled is not None:
                # Filtering cached/crowd-only images and crop retries also vary this batch.
                bucket = min(self.image_batch_size, 1 << (len(images) - 1).bit_length())
                encoder_images = [*images, *([images[-1]] * (bucket - len(images)))]
            pixel_values = self.processor(images=encoder_images, return_tensors="pt")["pixel_values"]
            if self.device.type == "cuda":
                pixel_values = pixel_values.pin_memory()
            pixel_values = pixel_values.to(self.device, dtype=self.dtype, non_blocking=self.device.type == "cuda")
            vision = self.encode_images(pixel_values=pixel_values)
            if self.encode_images.compiled is not None:
                # Own the features: subsequent CUDA Graph steps may overwrite encoder output buffers.
                vision = type(vision)(
                    fpn_hidden_states=tuple(level.clone() for level in vision.fpn_hidden_states),
                    fpn_position_encoding=tuple(level.clone() for level in vision.fpn_position_encoding),
                )
            del pixel_values
            jobs = ((index, annotation_id, box, text)
                    for index, image_prompts in enumerate(prompts) for annotation_id, box, text in image_prompts)
            for batch in batched(jobs, self.batch_size):
                real_count = len(batch)
                if self.decode_prompts.compiled is not None:
                    # A few power-of-two buckets avoid one compilation for every partial batch size.
                    bucket = min(self.batch_size, 1 << (real_count - 1).bit_length())
                    batch += (batch[-1],) * (bucket - real_count)
                indices = torch.tensor([job[0] for job in batch], device=self.device)
                # The encoder runs once per image, even for different classes/instances.
                features = type(vision)(
                    fpn_hidden_states=tuple(level.index_select(0, indices) for level in vision.fpn_hidden_states),
                    fpn_position_encoding=tuple(level.index_select(0, indices) for level in vision.fpn_position_encoding),
                )
                text_embeds, attention_mask = self.text_features([job[3] for job in batch])
                boxes = []
                for index, _, (x1, y1, x2, y2), _ in batch:
                    width, height = images[index].size
                    boxes.append([[(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                                   (x2 - x1) / width, (y2 - y1) / height]])
                pred_masks, pred_logits = self.decode_prompts(
                    vision_embeds=features, text_embeds=text_embeds, attention_mask=attention_mask,
                    input_boxes=torch.tensor(boxes, dtype=torch.float32, device=self.device),
                    input_boxes_labels=torch.ones(len(batch), 1, dtype=torch.long, device=self.device),
                )
                if pred_masks.ndim != 4 or pred_masks.shape[0] != len(batch):
                    raise RuntimeError(f"Unexpected SAM3 mask shape: {pred_masks.shape}")
                for offset, (index, annotation_id, box, _) in enumerate(batch[:real_count]):
                    size = (images[index].height, images[index].width)
                    mask = select_mask(pred_masks[offset], pred_logits[offset], box, size, torch)
                    yield index, annotation_id, mask
                del pred_masks, pred_logits, features, text_embeds, attention_mask


def encode_mask(mask):
    import numpy as np
    from pycocotools import mask as mask_utils

    if mask is None or not mask.any():
        return {"discard_reason": "no finite nonempty SAM3 mask"}
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode("ascii")
    return {"segmentation": rle, "area": int(mask.sum())}


def retry_failed_masks(image, prompts, results, predict, batch_size):
    import numpy as np

    retries = []
    for annotation_id, (x1, y1, x2, y2), text in prompts:
        if "discard_reason" not in results[str(annotation_id)]:
            continue
        margin = max(16, max(x2 - x1, y2 - y1) / 2)
        crop = (max(0, floor(x1 - margin)), max(0, floor(y1 - margin)),
                min(image.width, ceil(x2 + margin)), min(image.height, ceil(y2 + margin)))
        left, top, _, _ = crop
        retries.append((annotation_id, [x1 - left, y1 - top, x2 - left, y2 - top], text, crop))
    for batch in batched(retries, batch_size):
        images = [image.crop(job[3]) for job in batch]
        crop_prompts = [[(annotation_id, box, text)] for annotation_id, box, text, _ in batch]
        for index, annotation_id, mask in predict(images, crop_prompts):
            key = str(annotation_id)
            if mask is None:
                results[key]["discard_reason"] += "; crop retry failed"
            else:
                left, top, right, bottom = batch[index][3]
                full_mask = np.pad(mask, ((top, image.height - bottom), (left, image.width - right)))
                results[key] = {**encode_mask(full_mask), "recovered_on_crop": True}


def prepare_batch(args, batch):
    pending = []
    reused = failures = 0
    errors = []
    for split, record, rows, names, cached_stamp in batch:
        source = args.input / split / record["file_name"]
        try:
            stamp = source_stamp(source)
            cached = stamp == cached_stamp
            image = read_image(source, record, pixels=not cached)
        except (OSError, ValueError) as exc:
            errors.append(f"{split} [{args.device}]: image ID {record['id']}: {exc}")
            failures += 1
            continue
        if cached:
            reused += 1
            continue
        prompts, results = [], {}
        for row in rows:
            if row.get("iscrowd", 0) == 1:
                # Rasterize on export, also when resuming checkpoints made before crowd handling.
                results[str(row["id"])] = {"crowd_bbox": True}
                continue
            box = prompt_box(row.get("bbox"), record["width"], record["height"])
            if box is None:
                results[str(row["id"])] = {"discard_reason": f"unusable box {row.get('bbox')!r}"}
            else:
                prompts.append((row["id"], box, names[row["category_id"]]))
        pending.append((split, record, source, stamp, image, prompts, results))
    return pending, reused, failures, errors


def process_prepared_batch(args, prepared, predict, notify):
    pending, reused, failures, errors = prepared
    for error in errors:
        notify(("error", error))
    if reused or failures:
        notify(("checked", (reused, failures)))
    with_prompts = [job for job in pending if job[5]]
    if with_prompts:
        for index, annotation_id, mask in predict([job[4] for job in with_prompts], [job[5] for job in with_prompts]):
            with_prompts[index][6][str(annotation_id)] = encode_mask(mask)
    for split, record, source, stamp, image, prompts, results in pending:
        retry_failed_masks(image, prompts, results, predict, args.image_batch_size)
        if source_stamp(source) != stamp:
            raise OSError(f"{source}: image changed during processing")
        notify(("result", (split, record["id"], stamp, dumps(results))))


def process_batch(args, batch, predict, notify):
    process_prepared_batch(args, prepare_batch(args, batch), predict, notify)


def prefetch_batches(args, batches):
    """Read/decode one batch ahead; the caller alone advances the SQLite iterator."""
    if not args.prefetch:
        for batch in batches:
            yield prepare_batch(args, batch)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="SAM3 image reader") as pool:
        batch = next(batches, None)
        future = pool.submit(prepare_batch, args, batch) if batch is not None else None
        while future is not None:
            prepared = future.result()
            batch = next(batches, None)
            future = pool.submit(prepare_batch, args, batch) if batch is not None else None
            yield prepared


def lazy_predict(args):
    predictor = None

    def predict(images, prompts):
        nonlocal predictor
        if predictor is None:
            predictor = Sam3(args)
        return predictor.predict(images, prompts)

    return predict


def iter_batches(index, progress_db, args):
    for split in SPLITS:
        metadata = json.loads(index.execute("SELECT metadata FROM splits WHERE name=?", (split,)).fetchone()[0])
        names = {category["id"]: category["name"] for category in metadata["categories"]}
        images = index.execute("SELECT image_id, payload FROM images WHERE split=? ORDER BY ordinal", (split,))
        for batch in batched(images, args.image_batch_size):
            jobs = []
            for image_id, payload in batch:
                rows = [json.loads(row[0]) for row in index.execute(
                    "SELECT payload FROM annotations WHERE split=? AND image_id=?", (split, image_id),
                )]
                cached = progress_db.execute("SELECT source_stamp FROM completed WHERE split=? AND image_id=?",
                                             (split, image_id)).fetchone()
                jobs.append((split, json.loads(payload), rows, names, cached[0] if cached else None))
            yield jobs


def infer_worker(args, pipe):
    import torch

    torch.set_num_threads(worker_cpu_threads(args.gpus))
    with pipe:
        predict = lazy_predict(args)
        pipe.send(("ready", None))
        if args.prefetch:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="SAM3 image reader") as pool:
                def receive():
                    batch = pipe.recv()
                    return prepare_batch(args, batch) if batch is not None else None

                prepared = receive()
                while prepared is not None:
                    pipe.send(("ready", None))
                    future = pool.submit(receive)
                    process_prepared_batch(args, prepared, predict, pipe.send)
                    prepared = future.result()
        else:
            while (batch := pipe.recv()) is not None:
                process_batch(args, batch, predict, pipe.send)
                pipe.send(("ready", None))
        pipe.send(("done", None))


def infer_on_gpus(args, batches, notify):
    """Dynamically distribute bounded batches to one spawned process per visible GPU."""
    context = mp.get_context("spawn")
    workers = {}
    try:
        for rank in range(args.gpus):
            worker_args = argparse.Namespace(**vars(args))
            worker_args.device = f"cuda:{rank}"
            reader, writer = context.Pipe()
            process = context.Process(name=f"SAM3 cuda:{rank}", target=infer_worker, args=(worker_args, writer))
            try:
                with writer:
                    process.start()
            except BaseException:
                reader.close()
                raise
            workers[reader] = process
        active = set(workers)
        while active:
            for pipe in wait(active):
                process = workers[pipe]
                try:
                    event = pipe.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f"{process.name} stopped before completing its batch. "
                                       "See the worker log and resume with the same output.") from exc
                if event[0] == "ready":
                    batch = next(batches, None)
                    pipe.send(batch)
                elif event[0] == "done":
                    process.join()
                    if process.exitcode != 0:
                        raise RuntimeError(f"{process.name} exited with status {process.exitcode}")
                    active.remove(pipe)
                else:
                    notify(event)
    finally:
        for process in workers.values():
            if process.is_alive():
                process.terminate()
        for pipe, process in workers.items():
            pipe.close()
            process.join()
            process.close()


def open_progress(db, identity):
    # One coordinator owns the checkpoint, regardless of how many GPUs are used.
    db.execute("PRAGMA locking_mode=EXCLUSIVE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("BEGIN IMMEDIATE")
    db.execute("CREATE TABLE IF NOT EXISTS run (identity TEXT NOT NULL)")
    db.execute("""CREATE TABLE IF NOT EXISTS completed (
        split TEXT, image_id INTEGER, source_stamp TEXT, masks TEXT, PRIMARY KEY (split, image_id)
    )""")
    encoded = json.dumps(identity, sort_keys=True)
    row = db.execute("SELECT identity FROM run").fetchone()
    if row is None:
        db.execute("INSERT INTO run VALUES (?)", (encoded,))
    elif row[0] != encoded:
        previous = json.loads(row[0])
        if "min_box_iou" not in previous:
            raise ValueError("Output belongs to a different input or mask configuration. Use a new --output directory.")
        previous.pop("min_box_iou")
        if previous != {**identity, "version": 1}:
            raise ValueError("Output belongs to a different input or mask configuration. Use a new --output directory.")
        # Successful full-image selections are unchanged. Retry the images whose old
        # cutoff triggered a rejection or crop, so the removed filter cannot persist.
        removed = db.execute("""DELETE FROM completed
            WHERE instr(masks, '"discard_reason":') > 0
               OR instr(masks, '"recovered_on_crop":') > 0
        """).rowcount
        db.execute("UPDATE run SET identity=?", (encoded,))
        print(f"Updated checkpoint without IoU filtering; {removed} images scheduled for recalculation", flush=True)
    db.commit()


def bbox_fallback(record, row):
    box = prompt_box(row.get("bbox"), record["width"], record["height"])
    if box is None:
        return None
    import numpy as np

    x1, y1, x2, y2 = box
    mask = np.zeros((record["height"], record["width"]), dtype=bool)
    mask[floor(y1):ceil(y2), floor(x1):ceil(x2)] = True
    return encode_mask(mask)


def write_split(index, db, args, split, target, progress):
    from pycocotools import mask as mask_utils

    metadata = json.loads(index.execute("SELECT metadata FROM splits WHERE name=?", (split,)).fetchone()[0])
    names = {row["id"]: row["name"] for row in metadata["categories"]}
    count = mask_count = kept = recovered = fake = crowd = invalid_boxes = 0
    with target.open("w", encoding="utf-8") as stream:
        stream.write(dumps(metadata)[:-1] + ',"annotations":[')
        for image_id, payload in index.execute(
            "SELECT image_id, payload FROM images WHERE split=? ORDER BY ordinal", (split,),
        ):
            record = json.loads(payload)
            source = args.input / split / record["file_name"]
            target_image = args.output / split / record["file_name"]
            stamp = source_stamp(source)
            saved = db.execute("SELECT masks FROM completed WHERE split=? AND image_id=? AND source_stamp=?",
                               (split, image_id, stamp)).fetchone()
            if saved is None:
                raise ValueError(f"{split}: image ID {image_id} has no complete checkpoint")
            results = json.loads(saved[0])
            rows = [json.loads(row[0]) for row in index.execute(
                "SELECT payload FROM annotations WHERE split=? AND image_id=?", (split, image_id),
            )]
            if results.keys() != {str(row["id"]) for row in rows}:
                raise ValueError(f"{split}: image ID {image_id} has incomplete annotation IDs")
            annotations = []
            for row in rows:
                result = results[str(row["id"])]
                is_crowd = row.get("iscrowd", 0) == 1
                if is_crowd or "discard_reason" in result:
                    context = (
                        f"split={split} image_id={image_id} file={record['file_name']!r} "
                        f"annotation_id={row['id']} category_id={row['category_id']} "
                        f"class={names[row['category_id']]!r}"
                    )
                    fallback = bbox_fallback(record, row)
                    if fallback is None:
                        tqdm.write(f"SKIP_INVALID_BBOX {context} iscrowd={row.get('iscrowd', 0)} "
                                   f"bbox={row.get('bbox')!r} reason=no_nonempty_box", file=sys.stderr)
                        invalid_boxes += 1
                        continue
                    if is_crowd:
                        tqdm.write(f"CROWD_BBOX_MASK {context} reason=iscrowd=1", file=sys.stderr)
                        crowd += 1
                    else:
                        tqdm.write(f"BBOX_MASK_FALLBACK {context} reason={result['discard_reason']}", file=sys.stderr)
                        fake += 1
                    result = fallback
                rle, area = result["segmentation"], result["area"]
                if (rle["size"] != [record["height"], record["width"]] or not isinstance(rle["counts"], str)
                        or not 0 < area <= record["height"] * record["width"] or mask_utils.area(rle) != area):
                    raise ValueError(f"{split}: invalid RLE or area for annotation ID {row['id']}")
                annotations.append({**row, "segmentation": rle, "area": area})
                recovered += result.get("recovered_on_crop", False)
            materialize_image(source, target_image, stamp)
            read_image(target_image, record, pixels=False)
            kept += 1
            for row in annotations:
                if count:
                    stream.write(",")
                stream.write(dumps(row))
                count += 1
                mask_count += 1
            progress.update()
        stream.write('],"images":[')
        separator = ""
        for (payload,) in index.execute(
            "SELECT payload FROM images WHERE split=? ORDER BY ordinal", (split,),
        ):
            stream.write(separator + payload)
            separator = ","
        stream.write("]}\n")
    return (f"{split}: {kept} images, {count} annotations, {mask_count} masks; "
            f"{fake} rectangular bbox fallback masks; "
            f"{crowd} rectangular crowd masks; "
            f"{invalid_boxes} skipped invalid boxes; "
            f"{recovered} recovered on crops")


def run(args):
    if args.gpus is not None:
        import torch

        visible = torch.cuda.device_count()
        if args.gpus > visible:
            raise ValueError(f"Requested {args.gpus} GPUs, but only {visible} CUDA devices are visible.")
        args.device = "cuda:0"
    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.input.is_relative_to(args.output) or args.output.is_relative_to(args.input):
        raise ValueError("--input and --output must be separate directory trees")
    digests, stamps = {}, {}
    for split in SPLITS:
        path = args.input / split / COCO_FILE
        stamps[split] = source_stamp(path)
        print(f"{split}: checking source annotations", flush=True)
        with path.open("rb") as stream:
            digests[split] = file_digest(stream, "sha256").hexdigest()
    progress_path = args.output / ".sam3-progress.sqlite3"
    if progress_path.is_symlink():
        raise ValueError(f"Refusing symlink at {progress_path}")
    for split in SPLITS:
        directory = args.output / split
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError(f"Expected a regular directory at {directory}")
        if not progress_path.exists() and directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Refusing to overwrite existing dataset at {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    # Check the run identity before touching or constructing the source index.
    identity = {"version": 2, "pipeline": "all-classes-bbox-and-text", "model": MODEL_ID,
                "input": str(args.input), "annotations": digests}
    interactive = sys.stderr.isatty()
    with closing(sqlite3.connect(progress_path, timeout=1)) as db:
        open_progress(db, identity)
        index_path = prepare_index(args, digests)
        for split in SPLITS:
            if source_stamp(args.input / split / COCO_FILE) != stamps[split]:
                raise OSError(f"{split}: source JSON changed while indexing")
        with closing(sqlite3.connect(index_path)) as index:
            total = index.execute("SELECT SUM(image_count) FROM splits").fetchone()[0]
            bar_options = {"total": total, "unit": "img", "dynamic_ncols": interactive,
                           "mininterval": 1 if interactive else 15}
            failures = 0
            description = f"SAM3 ({args.gpus} GPUs)" if args.gpus else "SAM3"
            with tqdm(desc=description, **bar_options) as progress:
                reused = 0

                def notify(event):
                    nonlocal reused, failures
                    kind, data = event
                    if kind == "error":
                        tqdm.write(data, file=sys.stderr)
                    elif kind == "checked":
                        cached, invalid = data
                        reused += cached
                        failures += invalid
                        progress.set_postfix(cached=reused, invalid=failures, refresh=False)
                        progress.update(cached + invalid)
                    elif kind == "result":
                        with db:
                            db.execute("INSERT OR REPLACE INTO completed VALUES (?, ?, ?, ?)", data)
                        progress.update()
                    else:
                        raise RuntimeError(f"Unknown worker event: {kind}")

                batches = iter_batches(index, db, args)
                if args.gpus and args.gpus > 1:
                    infer_on_gpus(args, batches, notify)
                else:
                    predict = lazy_predict(args)
                    for prepared in prefetch_batches(args, batches):
                        process_prepared_batch(args, prepared, predict, notify)
            if failures:
                raise ValueError(f"{failures} unreadable/invalid images. Coverage is incomplete; fix and resume.")
            for split in SPLITS:
                if source_stamp(args.input / split / COCO_FILE) != stamps[split]:
                    raise OSError(f"{split}: source JSON changed during inference")
            summaries = []
            with tqdm(desc="COCO export", **bar_options) as progress, ExitStack() as stack:
                for split in SPLITS:
                    target = stack.enter_context(atomic_output(args.output / split / COCO_FILE))
                    summaries.append(write_split(index, db, args, split, target, progress))
            for summary in summaries:
                print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Requires Python 3.12+, the sam dependency group and Hugging Face access to facebook/sam3. "
        "Uses the existing annotation bbox AND its category name; preserves original IDs, classes and bboxes. "
        "Crowd annotations (iscrowd=1) use rectangular bbox masks without SAM3 and log each case. "
        "Failed SAM3 masks automatically use rectangular bbox masks after crop retry, with a log for each fallback. "
        "Annotations with unusable boxes that cannot produce a fallback mask are logged and omitted. "
        "Re-run with the same input/output to resume, including with a different number of GPUs.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Roboflow COCO root containing train/ and valid/")
    parser.add_argument("--output", required=True, type=Path, help="Separate output directory")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", default="auto", help="PyTorch device (default: CUDA if available, otherwise CPU)")
    devices.add_argument("--gpus", type=int, help="Number of visible CUDA GPUs on this node (one process per GPU)")
    parser.add_argument("--batch-size", default=4, type=int, help="BBox + text requests per decoder pass, per GPU (default: 4)")
    parser.add_argument("--image-batch-size", default=1, type=int, help="Images per encoder pass, per GPU (default: 1)")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto",
                        help="Model/autocast precision; auto uses BF16 on supported CUDA GPUs, FP32 otherwise")
    parser.add_argument("--attn-implementation", choices=ATTENTION_BACKENDS, default="sdpa",
                        help="Attention backend (default: sdpa, using PyTorch kernels)")
    parser.add_argument("--compile", choices=("auto", "on", "off"), default="auto",
                        help="Compile encoder and prompt decoder; auto enables on Linux CUDA with Triton")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
                        default="reduce-overhead", help="torch.compile mode (default: reduce-overhead, with CUDA Graphs when supported)")
    parser.add_argument("--compile-scope", choices=("all", "encoder"), default="all",
                        help="Compile both stages or only the image encoder (default: all)")
    parser.add_argument("--compile-recompile-limit", type=int, default=128,
                        help="Dynamo variants per function before eager fallback (default: 128)")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow TF32 for remaining FP32 CUDA operations (default: enabled)")
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True,
                        help="Read/decode the next image batch during GPU inference (default: enabled)")
    args = parser.parse_args()
    if args.batch_size < 1 or args.image_batch_size < 1:
        parser.error("--batch-size and --image-batch-size must be positive")
    if args.gpus is not None and args.gpus < 1:
        parser.error("--gpus must be positive")
    if args.compile_recompile_limit < 1:
        parser.error("--compile-recompile-limit must be positive")
    try:
        run(args)
    except (ImportError, KeyError, OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved COCO instance masks for all classes to {args.output}")


if __name__ == "__main__":
    main()
