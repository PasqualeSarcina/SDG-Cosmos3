"""Augmentation settings and COCO metadata for evaluation."""

from copy import deepcopy

from rfdetr.datasets.aug_configs import AUG_AGGRESSIVE


def augmentation():
    config = deepcopy(AUG_AGGRESSIVE)
    del config["VerticalFlip"], config["Rotate"]
    config["Affine"]["rotate"] = (-15, 15)
    return config


def split_coco(datamodule, split):
    return getattr(datamodule, f"_dataset_{split}").coco


def split_metadata(coco):
    category = coco.label2cat[0]
    return {
        "category_id": category,
        "source_category_name": coco.cats[category]["name"],
        "model_label": 0,
        "images": len(coco.imgs),
        "negative_images": sum(not coco.imgToAnns.get(image_id) for image_id in coco.imgs),
    }
