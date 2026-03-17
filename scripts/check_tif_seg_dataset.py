#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MASK_SUFFIXES = {".tif", ".tiff", ".png", ".bmp"}


def read_raster(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"} and tifffile is not None:
        return tifffile.imread(path)
    with Image.open(path) as image:
        return np.array(image)


def normalize_mask_array(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")
    return mask


def collect_files(root: str, suffixes: set[str]) -> dict[str, str]:
    root_path = Path(root)
    mapping = {}
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel_key = str(path.relative_to(root_path).with_suffix("")).replace("\\", "/")
        mapping[rel_key] = str(path)
    return mapping


def resolve_split_dirs(args, split: str) -> tuple[str, str]:
    if split == "train":
        image_dir = args.train_images or os.path.join(args.root, "train", "images")
        mask_dir = args.train_masks or os.path.join(args.root, "train", "masks")
    else:
        image_dir = args.val_images or os.path.join(args.root, "val", "images")
        mask_dir = args.val_masks or os.path.join(args.root, "val", "masks")
    return image_dir, mask_dir


def sha1_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha1_pair(image_path: str, mask_path: str) -> str:
    digest = hashlib.sha1()
    digest.update(sha1_file(image_path).encode("ascii"))
    digest.update(b"|")
    digest.update(sha1_file(mask_path).encode("ascii"))
    return digest.hexdigest()


def summarize_split(name: str, image_dir: str, mask_dir: str, label_values: tuple[int, ...]):
    image_files = collect_files(image_dir, IMAGE_SUFFIXES)
    mask_files = collect_files(mask_dir, MASK_SUFFIXES)
    shared_keys = sorted(set(image_files) & set(mask_files))
    image_only = sorted(set(image_files) - set(mask_files))
    mask_only = sorted(set(mask_files) - set(image_files))

    pixel_counter = Counter()
    file_presence_counter = Counter()
    unknown_value_counter = Counter()
    single_class_files = []
    example_values = {}

    for key in shared_keys:
        mask = normalize_mask_array(read_raster(mask_files[key]))
        unique_values, counts = np.unique(mask, return_counts=True)
        example_values[key] = [int(v) for v in unique_values.tolist()]
        if len(unique_values) == 1:
            single_class_files.append(key)
        for raw_value, count in zip(unique_values.tolist(), counts.tolist()):
            raw_value = int(raw_value)
            count = int(count)
            pixel_counter[raw_value] += count
            file_presence_counter[raw_value] += 1
            if raw_value not in label_values:
                unknown_value_counter[raw_value] += count

    total_pixels = sum(pixel_counter.values())
    labeled_pixels = sum(pixel_counter[v] for v in label_values)
    return {
        "name": name,
        "image_dir": str(Path(image_dir).resolve()),
        "mask_dir": str(Path(mask_dir).resolve()),
        "n_images": len(image_files),
        "n_masks": len(mask_files),
        "n_pairs": len(shared_keys),
        "image_only": image_only,
        "mask_only": mask_only,
        "shared_keys": shared_keys,
        "pixel_counter": pixel_counter,
        "file_presence_counter": file_presence_counter,
        "unknown_value_counter": unknown_value_counter,
        "single_class_files": single_class_files,
        "total_pixels": total_pixels,
        "labeled_pixels": labeled_pixels,
        "example_values": example_values,
    }


def print_split_summary(summary, label_values: tuple[int, ...], sample_limit: int):
    print(f"\n[{summary['name']}]")
    print(f"image_dir: {summary['image_dir']}")
    print(f"mask_dir: {summary['mask_dir']}")
    print(f"images={summary['n_images']} masks={summary['n_masks']} matched_pairs={summary['n_pairs']}")

    if summary["image_only"]:
        print(f"image files without masks: {len(summary['image_only'])}")
        print(f"sample image-only keys: {summary['image_only'][:sample_limit]}")
    if summary["mask_only"]:
        print(f"mask files without images: {len(summary['mask_only'])}")
        print(f"sample mask-only keys: {summary['mask_only'][:sample_limit]}")

    if summary["n_pairs"] == 0:
        print("no matched image/mask pairs found")
        return

    print("raw mask values by pixel count:")
    for raw_value, count in sorted(summary["pixel_counter"].items()):
        ratio = 100.0 * count / max(summary["total_pixels"], 1)
        marker = "" if raw_value in label_values else "  <-- not in label_values"
        print(f"  value={raw_value:<6} pixels={count:<12} ratio={ratio:6.2f}%{marker}")

    if summary["unknown_value_counter"]:
        unknown_pixels = sum(summary["unknown_value_counter"].values())
        ratio = 100.0 * unknown_pixels / max(summary["total_pixels"], 1)
        print(f"unknown-label pixels: {unknown_pixels} ({ratio:.2f}%)")
    else:
        print("unknown-label pixels: 0")

    single_class_ratio = 100.0 * len(summary["single_class_files"]) / max(summary["n_pairs"], 1)
    print(f"single-class masks: {len(summary['single_class_files'])}/{summary['n_pairs']} ({single_class_ratio:.2f}%)")
    if summary["single_class_files"]:
        print(f"sample single-class files: {summary['single_class_files'][:sample_limit]}")

    print("file presence per raw value:")
    for raw_value, count in sorted(summary["file_presence_counter"].items()):
        ratio = 100.0 * count / max(summary["n_pairs"], 1)
        print(f"  value={raw_value:<6} files={count:<8} ratio={ratio:6.2f}%")

    sample_keys = summary["shared_keys"][:sample_limit]
    if sample_keys:
        print("sample pair values:")
        for key in sample_keys:
            print(f"  {key}: {summary['example_values'][key]}")


def compare_splits(train_summary, val_summary, sample_limit: int):
    train_keys = set(train_summary["shared_keys"])
    val_keys = set(val_summary["shared_keys"])
    overlap_keys = sorted(train_keys & val_keys)

    print("\n[split comparison]")
    print(f"train/val key overlap: {len(overlap_keys)}")
    if overlap_keys:
        print(f"sample overlapping keys: {overlap_keys[:sample_limit]}")

    if train_summary["image_dir"] == val_summary["image_dir"]:
        print("WARNING: train_images and val_images resolve to the same directory")
    if train_summary["mask_dir"] == val_summary["mask_dir"]:
        print("WARNING: train_masks and val_masks resolve to the same directory")


def hash_leak_check(train_summary, val_summary, sample_limit: int):
    print("\n[hash leak check]")
    print("computing pair hashes, this may take a while...")

    train_hashes = {}
    for key in train_summary["shared_keys"]:
        image_path = os.path.join(train_summary["image_dir"], key)  # placeholder, replaced below
        del image_path

    train_hash_to_key = {}
    for key in train_summary["shared_keys"]:
        image_path = find_path_from_key(train_summary["image_dir"], key, IMAGE_SUFFIXES)
        mask_path = find_path_from_key(train_summary["mask_dir"], key, MASK_SUFFIXES)
        pair_hash = sha1_pair(image_path, mask_path)
        train_hash_to_key[pair_hash] = key

    leaked = []
    for key in val_summary["shared_keys"]:
        image_path = find_path_from_key(val_summary["image_dir"], key, IMAGE_SUFFIXES)
        mask_path = find_path_from_key(val_summary["mask_dir"], key, MASK_SUFFIXES)
        pair_hash = sha1_pair(image_path, mask_path)
        if pair_hash in train_hash_to_key:
            leaked.append((key, train_hash_to_key[pair_hash]))

    print(f"exact duplicated train/val pairs by content hash: {len(leaked)}")
    if leaked:
        print(f"sample duplicated pairs: {leaked[:sample_limit]}")


def find_path_from_key(root_dir: str, key: str, suffixes: set[str]) -> str:
    root = Path(root_dir)
    for suffix in suffixes:
        candidate = root / f"{key}{suffix}"
        if candidate.exists():
            return str(candidate)
    matches = list(root.glob(f"{key}.*"))
    if len(matches) == 1:
        return str(matches[0])
    raise FileNotFoundError(f"Could not uniquely resolve key '{key}' under {root_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Check a TIFF/PNG segmentation dataset for train/val leakage and label issues.")
    parser.add_argument("--root", required=True, help="Dataset root containing train/ and val/ by default.")
    parser.add_argument("--train-images", default=None, help="Override train image directory.")
    parser.add_argument("--train-masks", default=None, help="Override train mask directory.")
    parser.add_argument("--val-images", default=None, help="Override val image directory.")
    parser.add_argument("--val-masks", default=None, help="Override val mask directory.")
    parser.add_argument("--label-values", nargs="*", type=int, default=[0, 80, 160, 240], help="Expected raw mask values.")
    parser.add_argument("--sample-limit", type=int, default=10, help="How many examples to print per section.")
    parser.add_argument("--hash-check", action="store_true", help="Also detect exact duplicated train/val pairs by file content.")
    parser.add_argument("--json", action="store_true", help="Print a compact JSON summary at the end.")
    return parser.parse_args()


def main():
    args = parse_args()
    label_values = tuple(int(v) for v in args.label_values)

    train_image_dir, train_mask_dir = resolve_split_dirs(args, "train")
    val_image_dir, val_mask_dir = resolve_split_dirs(args, "val")

    train_summary = summarize_split("train", train_image_dir, train_mask_dir, label_values)
    val_summary = summarize_split("val", val_image_dir, val_mask_dir, label_values)

    print_split_summary(train_summary, label_values, args.sample_limit)
    print_split_summary(val_summary, label_values, args.sample_limit)
    compare_splits(train_summary, val_summary, args.sample_limit)

    if args.hash_check:
        hash_leak_check(train_summary, val_summary, args.sample_limit)

    if args.json:
        payload = {
            "train": {
                "n_images": train_summary["n_images"],
                "n_masks": train_summary["n_masks"],
                "n_pairs": train_summary["n_pairs"],
                "single_class_files": len(train_summary["single_class_files"]),
                "pixel_counter": dict(sorted(train_summary["pixel_counter"].items())),
                "unknown_value_counter": dict(sorted(train_summary["unknown_value_counter"].items())),
            },
            "val": {
                "n_images": val_summary["n_images"],
                "n_masks": val_summary["n_masks"],
                "n_pairs": val_summary["n_pairs"],
                "single_class_files": len(val_summary["single_class_files"]),
                "pixel_counter": dict(sorted(val_summary["pixel_counter"].items())),
                "unknown_value_counter": dict(sorted(val_summary["unknown_value_counter"].items())),
            },
        }
        print("\n[json]")
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
