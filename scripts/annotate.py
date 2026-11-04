"""Generate COCO bounding boxes and instance masks for passenger cars using SAM3."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MODEL_ID = "facebook/sam3"
CATEGORY = {"id": 1, "name": "passenger car"}


def collect_images(source: Path) -> tuple[Path, list[Path]]:
    if source.is_dir():
        images = sorted(
            path for path in source.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
        )
        if images:
            return source, images
        raise ValueError(f"No supported images found in {source}")
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return source.parent, [source]
    raise ValueError(f"Expected an image or an image directory: {source}")


def coco_masks(masks, mask_utils):
    import numpy as np

    for mask in masks.cpu().numpy():
        rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
        area = int(mask_utils.area(rle))
        if area:
            bbox = mask_utils.toBbox(rle).tolist()
            rle["counts"] = rle["counts"].decode("ascii")
            yield {"bbox": bbox, "area": area, "segmentation": rle}


def annotate(root: Path, paths: list[Path]) -> dict:
    import torch
    from transformers import Sam3Model, Sam3Processor

    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("Install the SAM3 dependencies with 'uv sync --group sam'") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID).to(device).eval()
    dataset = {
        "info": {"description": "SAM3 pseudo-annotations"},
        "images": [],
        "annotations": [],
        "categories": [CATEGORY.copy()],
    }
    with torch.inference_mode():
        text_inputs = processor(text=CATEGORY["name"], return_tensors="pt").to(device)
        for image_id, path in enumerate(tqdm(paths, unit="image"), start=1):
            try:
                with Image.open(path) as source:
                    image = source.convert("RGB")
            except OSError as exc:
                raise OSError(f"Cannot read {path}: {exc}") from exc
            width, height = image.size
            dataset["images"].append({
                "id": image_id,
                "file_name": path.relative_to(root).as_posix(),
                "width": width,
                "height": height,
            })
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            vision_embeds = model.get_vision_features(pixel_values=pixel_values)
            outputs = model(vision_embeds=vision_embeds, **text_inputs)
            result = processor.post_process_instance_segmentation(
                outputs, threshold=0.5, mask_threshold=0.5, target_sizes=[(height, width)],
            )[0]
            for instance in coco_masks(result["masks"].to(dtype=torch.uint8), mask_utils):
                dataset["annotations"].append({
                    "id": len(dataset["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": CATEGORY["id"],
                    "iscrowd": 0,
                    **instance,
                })
    return dataset


def save_coco(output: Path, dataset: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(dataset, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Requires PyTorch with torchvision, Transformers 5.x and pycocotools. "
        "The facebook/sam3 checkpoint requires Hugging Face access and login.",
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Image file or directory searched recursively")
    parser.add_argument("-o", "--output", type=Path, required=True, help="COCO JSON file (replaced if it exists)")
    args = parser.parse_args()

    if args.output.suffix.lower() != ".json" or args.output.is_dir():
        parser.error("--output must be a .json file")
    try:
        root, paths = collect_images(args.input)
        dataset = annotate(root, paths)
        save_coco(args.output, dataset)
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {len(dataset['annotations'])} annotations for {len(dataset['images'])} images to {args.output}")


if __name__ == "__main__":
    main()
