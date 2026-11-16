# /// script
# requires-python = ">=3.10"
# dependencies = ["ijson>=3.2", "pillow"]
# ///
"""Download all Objects365 v2 classes in Roboflow COCO detection format.

Run with: uv run --script scripts/download_objects365.py --output /data/objects365
Standalone script; requires only ijson and Pillow, no companion project files.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, suppress
from http.client import HTTPException
import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tarfile
from tempfile import NamedTemporaryFile
from threading import Event
from time import sleep
from urllib.request import urlopen

from PIL import Image


# Source constants and transfer helpers are kept here so this file runs on its own.
# The official training JSON lists three JPEGs absent from the image archives:
# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/objects365.py
BASE_URL = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86"


SPLITS = {"train": ("train", 1_742_292), "val": ("valid", 80_000)}


ANNOTATION_FILES = {"train": "zhiyuan_objv2_train.tar.gz", "val": "zhiyuan_objv2_val.json"}


KNOWN_MISSING_TRAIN = {
    "objects365_v2_00908726.jpg",
    "objects365_v1_00320532.jpg",
    "objects365_v1_00320534.jpg",
}


BLOCK_SIZE = 1024 * 1024


MIN_FREE_BYTES = 5_000_000_000


class InvalidImageError(ValueError):
    """The source image cannot be used with its official annotation dimensions."""


@contextmanager
def atomic_output(path):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(path)
    except BaseException:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


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


def valid_jpeg(path, dimensions):
    try:
        with Image.open(path) as image:
            return image.format == "JPEG" and image.size == dimensions
    except FileNotFoundError:
        return False
    except OSError as exc:
        if exc.errno is None:
            return False
        raise RuntimeError(f"Cannot read JPEG on output filesystem at {path}: {exc}") from exc


def save_jpeg(path, jpeg, size, dimensions):
    if len(jpeg) != size:
        raise EOFError(f"Incomplete JPEG: {path.name}")
    try:
        check_space(path.parent, size)
        with atomic_output(path) as temporary:
            temporary.write_bytes(jpeg)
            try:
                with Image.open(temporary) as image:
                    if image.format != "JPEG" or image.size != dimensions:
                        raise InvalidImageError(
                            f"Invalid image: {path.name}: expected JPEG {dimensions[0]}x{dimensions[1]}, "
                            f"found {image.format} {image.width}x{image.height}"
                        )
            except OSError as exc:
                if exc.errno is not None:
                    raise
                raise InvalidImageError(
                    f"Unreadable image: {path.name}: expected JPEG {dimensions[0]}x{dimensions[1]}: {exc}"
                ) from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot write JPEG on output filesystem at {path}: {exc}") from exc


def download_patch(relative_url, selected, output, stop, *, invalid_images=None):
    """Optionally record and exclude image validation failures after three attempts.

    invalid_images must be a separate dict for each archive. Network, archive,
    incomplete-transfer and filesystem errors are never treated as invalid images.
    """
    pending = {name: size for name, size in selected.items() if not valid_jpeg(output / name, size)}
    if not pending:
        print(f"{relative_url}: all {len(selected):,} JPEGs already present", flush=True)
        return set()
    print(f"{relative_url}: looking for {len(pending):,} JPEGs", flush=True)
    check_space(output)
    for attempt in range(3):
        if stop.is_set():
            return set()
        try:
            with urlopen(f"{BASE_URL}/{relative_url}", timeout=60) as response:
                with tarfile.open(fileobj=response, mode="r|gz", bufsize=BLOCK_SIZE) as archive:
                    for member in archive:
                        if stop.is_set():
                            return set()
                        name = PurePosixPath(member.name).name
                        if not member.isfile() or name not in pending:
                            continue
                        with archive.extractfile(member) as source:
                            jpeg = source.read()
                        try:
                            save_jpeg(output / name, jpeg, member.size, pending[name])
                        except InvalidImageError as exc:
                            if invalid_images is None:
                                raise
                            if attempt < 2:
                                print(f"{relative_url}: will retry {exc}", flush=True)
                                continue
                            invalid_images[name] = {"archive": relative_url, "error": str(exc)}
                            print(f"{relative_url}: excluding after 3 attempts: {exc}", flush=True)
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


JSON_OPTIONS = {"separators": (",", ":"), "allow_nan": False}


def read_categories(path, split, ijson):
    with open_annotations(path, split) as stream:
        categories = sorted(ijson.items(stream, "categories.item"), key=lambda row: row["id"])
    if len(categories) != 365 or len({row["id"] for row in categories}) != 365:
        raise ValueError(f"{split}: expected 365 distinct categories")
    return categories


def build_image_index(path, split, root, ijson):
    """Index image metadata on disk; never load the annotation array into RAM."""
    destination = root / ".objects365-source" / f"{split}-images.sqlite"
    with atomic_output(destination) as temporary:
        db = sqlite3.connect(temporary)
        try:
            db.execute("""CREATE TABLE images (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                patch TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                payload TEXT NOT NULL
            )""")

            def rows():
                with open_annotations(path, split) as stream:
                    for image in ijson.items(stream, "images.item", use_float=True):
                        filename = image["file_name"]
                        patch = archive_path(split, filename)
                        name = PurePosixPath(filename).name
                        if type(image["id"]) is not int or any(
                            type(image[key]) is not int or image[key] < 1 for key in ("width", "height")
                        ):
                            raise ValueError(f"{split}: invalid image ID or dimensions: {filename}")
                        image["objects365_file_name"] = filename
                        image["file_name"] = name
                        yield (
                            image["id"], name, patch, image["width"], image["height"],
                            json.dumps(image, **JSON_OPTIONS),
                        )

            with db:
                db.executemany("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?)", rows())
                count = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
                expected = SPLITS[split][1]
                if count != expected:
                    raise ValueError(f"{split}: expected {expected:,} official images, found {count:,}")
                db.execute("CREATE INDEX images_patch ON images (patch)")
        finally:
            db.close()
    print(f"{split}: indexed {count:,} images", flush=True)
    return destination


def download_images(db, split, output, workers, skip_invalid_images=False):
    """Bound in-flight metadata to the number of active archive transfers."""
    patches = iter(row[0] for row in db.execute("SELECT DISTINCT patch FROM images ORDER BY patch"))
    missing = set()
    invalid = {}
    stop = Event()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {}

        def submit_next():
            patch = next(patches, None)
            if patch is None:
                return
            selected = {
                name: (width, height)
                for name, width, height in db.execute(
                    "SELECT name, width, height FROM images WHERE patch = ?", (patch,)
                )
            }
            rejected = {} if skip_invalid_images else None
            job = executor.submit(download_patch, patch, selected, output, stop, invalid_images=rejected)
            jobs[job] = rejected

        try:
            for _ in range(workers):
                submit_next()
            while jobs:
                completed, _ = wait(jobs, return_when=FIRST_COMPLETED)
                for job in completed:
                    rejected = jobs.pop(job)
                    missing.update(job.result())
                    if rejected:
                        invalid.update(rejected)
                    submit_next()
        except BaseException:
            stop.set()
            for job in jobs:
                job.cancel()
            raise
    allowed_missing = KNOWN_MISSING_TRAIN if split == "train" else set()
    if not missing <= allowed_missing:
        raise ValueError(f"{split}: unexpected missing images: {sorted(missing)}")
    return missing, invalid


def write_coco(path, split, db, categories, excluded, output, ijson):
    """Write a JSON record at a time, excluding unavailable images and their boxes."""
    image_ids = {
        image_id for image_id, name in db.execute("SELECT id, name FROM images") if name not in excluded
    }
    excluded_ids = {
        image_id for image_id, name in db.execute("SELECT id, name FROM images") if name in excluded
    }
    category_ids = {row["id"] for row in categories}
    counts = {"images": 0, "annotations": 0, "excluded_annotations": 0}
    destination = output / "_annotations.coco.json"
    with atomic_output(destination) as temporary, temporary.open("w", encoding="utf-8") as stream:
        info = {"description": "Objects365 v2, all 365 classes", "url": "https://www.objects365.org/"}
        stream.write('{"info":' + json.dumps(info, **JSON_OPTIONS))
        stream.write(',"licenses":[],"categories":' + json.dumps(categories, **JSON_OPTIONS))
        stream.write(',"images":[')
        for image_id, payload in db.execute("SELECT id, payload FROM images ORDER BY id"):
            if image_id not in image_ids:
                continue
            if counts["images"]:
                stream.write(",")
            stream.write(payload)
            counts["images"] += 1
            if counts["images"] % 10_000 == 0:
                check_space(output)
        stream.write('],"annotations":[')
        with open_annotations(path, split) as annotations:
            for row in ijson.items(annotations, "annotations.item", use_float=True):
                if row["image_id"] in excluded_ids:
                    counts["excluded_annotations"] += 1
                    continue
                if row["image_id"] not in image_ids or row["category_id"] not in category_ids:
                    raise ValueError(f"{split}: invalid image/category reference in annotation {row['id']}")
                if counts["annotations"]:
                    stream.write(",")
                stream.write(json.dumps(row, **JSON_OPTIONS))
                counts["annotations"] += 1
                if counts["annotations"] % 10_000 == 0:
                    check_space(output)
                if counts["annotations"] % 1_000_000 == 0:
                    print(f"{split}: wrote {counts['annotations']:,} boxes", flush=True)
        if counts["annotations"] + counts["excluded_annotations"] == 0:
            raise ValueError(f"{split}: source contains no annotations")
        stream.write("]}")
        check_space(output)
    return counts


def download_split(split, root, workers, ijson, expected_categories=None, *, skip_invalid_images=False):
    path = annotation_path(root, split)
    print(f"{split}: reading all 365 categories and image metadata", flush=True)
    categories = read_categories(path, split, ijson)
    if expected_categories is not None and categories != expected_categories:
        raise ValueError("Train and validation categories do not match")
    index = build_image_index(path, split, root, ijson)
    output = root / SPLITS[split][0]
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(index)
    try:
        missing, invalid = download_images(db, split, output, workers, skip_invalid_images)
        counts = write_coco(path, split, db, categories, missing | invalid.keys(), output, ijson)
    finally:
        db.close()
    summary = {
        "split": split, "categories": len(categories), **counts,
        "missing_images": sorted(missing), "invalid_images": invalid,
    }
    report = root / ".objects365-source" / f"{split}-download.json"
    with atomic_output(report) as temporary:
        temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{output}: {counts['images']:,} JPEGs, {counts['annotations']:,} boxes, 365 classes", flush=True)
    if missing:
        print(f"{split}: excluded documented unavailable JPEGs: {', '.join(sorted(missing))}", flush=True)
    if invalid:
        print(f"{split}: excluded {len(invalid):,} invalid images and their boxes; details in {report}", flush=True)
    return categories


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Saves train/ and valid/, with JPEGs and _annotations.coco.json in each. "
        "Image archives are streamed, not stored. Rerunning skips valid existing JPEGs. "
        "A partial archive must be read again from its beginning. Bounding boxes only.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Dataset output directory")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent image archive transfers (default: 4)")
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLITS), default=list(SPLITS))
    parser.add_argument(
        "--skip-invalid-images", action="store_true",
        help="After 3 attempts, exclude images with unreadable headers, non-JPEG format or unexpected "
        "dimensions, along with their boxes. Record details in .objects365-source/*-download.json.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        import ijson
    except ImportError:
        parser.exit(1, "Install dependencies: python -m pip install ijson pillow\n")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        check_space(args.output)
        print("Objects365 v2: ALL 365 classes; train + validation by default.", flush=True)
        print("Allow approximately 400 GB for the full extracted dataset, JSONs and image index.", flush=True)
        print("Resume with the same command; existing valid JPEGs will be reused.", flush=True)
        categories = None
        for split in dict.fromkeys(args.splits):
            categories = download_split(
                split, args.output, args.workers, ijson, categories,
                skip_invalid_images=args.skip_invalid_images,
            )
    except (ValueError, OSError, EOFError, RuntimeError, HTTPException,
            tarfile.TarError, sqlite3.Error, ijson.JSONError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted. Run the same command to resume.\n")


if __name__ == "__main__":
    main()
