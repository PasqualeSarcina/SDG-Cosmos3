"""Download the Objects365 v2 Car subset in Roboflow COCO detection format."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from http.client import HTTPException
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from tempfile import NamedTemporaryFile
from threading import Event
from time import sleep
from urllib.request import urlopen

from PIL import Image


BASE_URL = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86"
SPLITS = {"train": ("train", 1_742_292), "val": ("valid", 80_000)}
ANNOTATION_FILES = {"train": "zhiyuan_objv2_train.tar.gz", "val": "zhiyuan_objv2_val.json"}
# The official training JSON lists these JPEGs, which are absent from the image archives.
# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/objects365.py
KNOWN_MISSING_TRAIN = {
    "objects365_v2_00908726.jpg",
    "objects365_v1_00320532.jpg",
    "objects365_v1_00320534.jpg",
}
BLOCK_SIZE = 1024 * 1024
MIN_FREE_BYTES = 5_000_000_000


@contextmanager
def atomic_output(path):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def check_space(path, extra_bytes=0):
    if shutil.disk_usage(path).free < MIN_FREE_BYTES + extra_bytes:
        raise RuntimeError(f"Not enough space in {path}. The downloader keeps 5 GB free.")


def annotation_path(output, split):
    cache = output / ".objects365-source"
    cache.mkdir(parents=True, exist_ok=True)
    filename = ANNOTATION_FILES[split]
    path = cache / filename
    if path.is_file():
        return path

    print(f"Downloading {filename}", flush=True)
    for attempt in range(3):
        try:
            with urlopen(f"{BASE_URL}/{split}/{filename}", timeout=60) as response:
                length = response.headers.get("Content-Length")
                check_space(cache, int(length or 0))
                with atomic_output(path) as temporary, temporary.open("wb") as stream:
                    while block := response.read(BLOCK_SIZE):
                        check_space(cache, len(block))
                        stream.write(block)
                    if length is not None and stream.tell() != int(length):
                        raise EOFError(f"Incomplete download: {filename}")
            return path
        except (OSError, EOFError, HTTPException):
            if attempt == 2:
                raise
            sleep(2 ** attempt)


@contextmanager
def open_annotations(path, split):
    if split == "val":
        with path.open("rb") as stream:
            yield stream
        return
    expected = "zhiyuan_objv2_train.json"
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.isfile() and PurePosixPath(member.name).name == expected:
                with archive.extractfile(member) as stream:
                    yield stream
                return
    raise ValueError(f"{expected} was not found in {path}")


def archive_path(split, filename):
    path = PurePosixPath(filename)
    if len(path.parts) != 4 or path.parts[0] != "images" or path.suffix.lower() != ".jpg":
        raise ValueError(f"Unexpected Objects365 image path: {filename}")
    _, version, patch, _ = path.parts
    if version not in {"v1", "v2"} or not patch.startswith("patch") or not patch[5:].isdigit():
        raise ValueError(f"Unexpected Objects365 patch: {filename}")
    if split == "train":
        return f"train/{patch}.tar.gz"
    return f"val/images/{version}/{patch}.tar.gz"


def read_split(path, split, ijson):
    _, expected = SPLITS[split]
    print(f"Checking {split} annotations in {path}", flush=True)
    with open_annotations(path, split) as stream:
        count = sum(1 for _ in ijson.items(stream, "images.item"))
    if count != expected:
        raise ValueError(f"{split}: expected {expected:,} official image records, found {count:,} in {path}")

    with open_annotations(path, split) as stream:
        categories = list(ijson.items(stream, "categories.item"))
    car = [category for category in categories if category["name"].strip().casefold() == "car"]
    if len(categories) != 365 or len(car) != 1:
        raise ValueError(f"{split}: expected 365 categories including one Car category")
    car = car[0]
    with open_annotations(path, split) as stream:
        annotations = [
            row for row in ijson.items(stream, "annotations.item", use_float=True)
            if row["category_id"] == car["id"]
        ]
    image_ids = {row["image_id"] for row in annotations}
    if not image_ids:
        raise ValueError(f"{split}: no Car annotations found in {path}")

    images = []
    patches = defaultdict(dict)
    names = set()
    with open_annotations(path, split) as stream:
        for image in ijson.items(stream, "images.item"):
            if image["id"] not in image_ids:
                continue
            source = image["file_name"]
            name = PurePosixPath(source).name
            if name in names:
                raise ValueError(f"{split}: duplicate Car filename {name}")
            names.add(name)
            patches[archive_path(split, source)][name] = (image["width"], image["height"])
            images.append({
                "id": image["id"],
                "file_name": name,
                "width": image["width"],
                "height": image["height"],
                "objects365_file_name": source,
            })
    if len(images) != len(image_ids) or {image["id"] for image in images} != image_ids:
        raise ValueError(f"{split}: duplicate or missing image records for Car annotations")
    print(f"{split}: {count:,} official records, {len(images):,} Car images, {len(annotations):,} boxes", flush=True)
    return images, annotations, car, patches


def valid_jpeg(path, dimensions):
    try:
        with Image.open(path) as image:
            return image.format == "JPEG" and image.size == dimensions
    except OSError:
        return False


def download_patch(relative_url, selected, output, stop):
    pending = {name: size for name, size in selected.items() if not valid_jpeg(output / name, size)}
    if not pending:
        print(f"{relative_url}: all {len(selected):,} JPEGs already present", flush=True)
        return set()
    print(f"{relative_url}: looking for {len(pending):,} JPEGs", flush=True)
    for attempt in range(3):
        if stop.is_set():
            return set()
        try:
            check_space(output)
            with urlopen(f"{BASE_URL}/{relative_url}", timeout=60) as response:
                with tarfile.open(fileobj=response, mode="r|gz", bufsize=BLOCK_SIZE) as archive:
                    for member in archive:
                        if stop.is_set():
                            return set()
                        name = PurePosixPath(member.name).name
                        if not member.isfile() or name not in pending:
                            continue
                        check_space(output, member.size)
                        with atomic_output(output / name) as temporary:
                            with archive.extractfile(member) as source, temporary.open("wb") as target:
                                shutil.copyfileobj(source, target, BLOCK_SIZE)
                            if temporary.stat().st_size != member.size or not valid_jpeg(temporary, pending[name]):
                                raise ValueError(f"Invalid JPEG or unexpected dimensions: {name}")
                        del pending[name]
                        if not pending:
                            break
            if not pending or (relative_url.startswith("train/") and pending.keys() <= KNOWN_MISSING_TRAIN):
                print(f"{relative_url}: {len(selected) - len(pending):,}/{len(selected):,} JPEGs ready", flush=True)
                return set(pending)
            raise EOFError(f"{relative_url}: {len(pending):,} expected JPEGs absent, e.g. {next(iter(pending))}")
        except (OSError, EOFError, HTTPException, tarfile.TarError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"{relative_url}: {exc}") from exc
            print(f"{relative_url}: retrying after {exc}", flush=True)
            sleep(2 ** attempt)


def download_split(split, plan, root, workers):
    images, annotations, category, patches = plan
    output = root / SPLITS[split][0]
    output.mkdir(parents=True, exist_ok=True)
    missing = set()
    stop = Event()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = [executor.submit(download_patch, url, names, output, stop) for url, names in patches.items()]
        try:
            for job in as_completed(jobs):
                missing.update(job.result())
        except BaseException:
            stop.set()
            for job in jobs:
                job.cancel()
            raise

    if missing:
        print(f"{split}: excluding documented unavailable JPEGs: {', '.join(sorted(missing))}", flush=True)
        images = [image for image in images if image["file_name"] not in missing]
        ids = {image["id"] for image in images}
        annotations = [row for row in annotations if row["image_id"] in ids]
    coco = {
        "info": {"description": "Objects365 v2 Car bounding boxes", "url": "https://www.objects365.org/"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [category],
    }
    path = output / "_annotations.coco.json"
    check_space(output)
    with atomic_output(path) as temporary, temporary.open("w", encoding="utf-8") as stream:
        json.dump(coco, stream, separators=(",", ":"), allow_nan=False)
    print(f"{output}: {len(images):,} JPEGs, {len(annotations):,} boxes", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Official annotation files are cached in OUTPUT/.objects365-source. Image archives "
        "are streamed and only selected JPEGs are written. Requires ijson and Pillow.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory containing train/ and valid/")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent image archive transfers (default: 4)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    try:
        import ijson
    except ImportError:
        parser.exit(1, "Install ijson with 'python -m pip install ijson' or run with 'uv run --with ijson'.\n")

    try:
        # Check both source splits before starting any image transfer.
        plans = {
            split: read_split(annotation_path(args.output, split), split, ijson)
            for split in ("val", "train")
        }
        for split in SPLITS:
            download_split(split, plans[split], args.output, args.workers)
    except (ValueError, OSError, EOFError, RuntimeError, HTTPException, tarfile.TarError, ijson.JSONError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
