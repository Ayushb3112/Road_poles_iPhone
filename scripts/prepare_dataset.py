#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def read_split_file(path: Path) -> List[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean:
            lines.append(clean)
    return lines


def normalize_image_rel_path(rel_path: str) -> str:
    clean = rel_path.strip().replace("\\", "/")
    if clean.startswith("./"):
        clean = clean[2:]
    if clean.startswith("data/"):
        clean = clean[len("data/") :]
    return clean


def image_to_label_path(image_rel_path: str) -> str:
    if image_rel_path.startswith("images/"):
        label_rel = "labels/" + image_rel_path[len("images/") :]
    else:
        label_rel = image_rel_path.replace("/images/", "/labels/")
    return str(Path(label_rel).with_suffix(".txt")).replace("\\", "/")


def parse_label_line(line: str) -> Tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Label row should have 5 values, got {len(parts)}: {line}")
    cls = int(parts[0])
    vals = [float(x) for x in parts[1:]]
    return cls, vals[0], vals[1], vals[2], vals[3]


def validate_label_file(label_path: Path) -> Dict[str, int]:
    result = {"boxes": 0, "invalid_rows": 0, "invalid_class": 0, "invalid_box": 0}
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        result["boxes"] += 1
        try:
            cls, x, y, w, h = parse_label_line(line)
        except Exception:
            result["invalid_rows"] += 1
            continue
        if cls != 0:
            result["invalid_class"] += 1
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            result["invalid_box"] += 1
    return result


def process_split(
    root: Path,
    split_name: str,
    split_file: Path,
    output_split_file: Path,
    allow_missing_labels: bool,
) -> Dict[str, object]:
    original_lines = read_split_file(split_file)
    normalized_paths: List[str] = []
    missing_images: List[str] = []
    missing_labels: List[str] = []

    boxes = invalid_rows = invalid_class = invalid_box = 0

    for item in original_lines:
        image_rel = normalize_image_rel_path(item)
        image_abs = root / image_rel
        if not image_abs.exists():
            missing_images.append(image_rel)
            continue
        normalized_paths.append(str(image_abs.resolve()))

        label_rel = image_to_label_path(image_rel)
        label_abs = root / label_rel
        if not label_abs.exists():
            if not allow_missing_labels:
                missing_labels.append(label_rel)
            continue

        stats = validate_label_file(label_abs)
        boxes += stats["boxes"]
        invalid_rows += stats["invalid_rows"]
        invalid_class += stats["invalid_class"]
        invalid_box += stats["invalid_box"]

    output_split_file.parent.mkdir(parents=True, exist_ok=True)
    output_split_file.write_text("\n".join(normalized_paths) + "\n", encoding="utf-8")

    return {
        "split": split_name,
        "total_listed": len(original_lines),
        "resolved_images": len(normalized_paths),
        "missing_images_count": len(missing_images),
        "missing_labels_count": len(missing_labels),
        "boxes_count": boxes,
        "invalid_rows_count": invalid_rows,
        "invalid_class_count": invalid_class,
        "invalid_box_count": invalid_box,
        "missing_images_examples": missing_images[:10],
        "missing_labels_examples": missing_labels[:10],
        "output_list_file": str(output_split_file.resolve()),
    }


def build_data_yaml(path: Path, train_list: Path, val_list: Path, test_list: Path) -> None:
    data = {
        "path": str(Path(".").resolve()),
        "train": str(train_list.resolve()),
        "val": str(val_list.resolve()),
        "test": str(test_list.resolve()),
        "names": {0: "pole"},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and validate Road Poles dataset.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root directory.")
    parser.add_argument(
        "--allow-missing-test-labels",
        action="store_true",
        help="Do not fail if test labels are absent (leaderboard setup).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    reports_dir = root / "reports"
    generated_dir = root / "generated"
    configs_dir = root / "configs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    train_report = process_split(
        root=root,
        split_name="train",
        split_file=root / "Train.txt",
        output_split_file=generated_dir / "train_abs.txt",
        allow_missing_labels=False,
    )
    val_report = process_split(
        root=root,
        split_name="val",
        split_file=root / "Validation.txt",
        output_split_file=generated_dir / "val_abs.txt",
        allow_missing_labels=False,
    )
    test_report = process_split(
        root=root,
        split_name="test",
        split_file=root / "Test.txt",
        output_split_file=generated_dir / "test_abs.txt",
        allow_missing_labels=args.allow_missing_test_labels,
    )

    data_yaml_path = configs_dir / "data_resolved.yaml"
    build_data_yaml(
        path=data_yaml_path,
        train_list=generated_dir / "train_abs.txt",
        val_list=generated_dir / "val_abs.txt",
        test_list=generated_dir / "test_abs.txt",
    )

    summary = {
        "dataset_root": str(root),
        "data_yaml": str(data_yaml_path.resolve()),
        "splits": {
            "train": train_report,
            "val": val_report,
            "test": test_report,
        },
    }

    report_path = reports_dir / "data_check.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote resolved YAML: {data_yaml_path}")
    print(f"Wrote data report:   {report_path}")
    for split in ("train", "val", "test"):
        r = summary["splits"][split]
        print(
            f"[{split}] listed={r['total_listed']} resolved={r['resolved_images']} "
            f"missing_images={r['missing_images_count']} missing_labels={r['missing_labels_count']}"
        )


if __name__ == "__main__":
    main()
