#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


def iter_test_images(test_list_file: Path) -> Iterable[Path]:
    for line in test_list_file.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean:
            yield Path(clean)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on test set and save YOLO-format predictions.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, default=Path("generated/test_abs.txt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_predictions_labels"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    image_paths = list(iter_test_images(args.test_list))
    if not image_paths:
        raise RuntimeError(f"No test images found in {args.test_list}")

    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=False,
        verbose=False,
    )

    for image_path, result in zip(image_paths, results):
        out_txt = args.output_dir / f"{image_path.stem}.txt"
        lines = []
        if result.boxes is not None and len(result.boxes) > 0:
            xywhn = result.boxes.xywhn.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy()
            for cls_id, conf, coords in zip(clses, confs, xywhn):
                x, y, w, h = coords.tolist()
                # Leaderboard expects: class x_center y_center width height confidence
                lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}")
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"Saved {len(image_paths)} prediction files to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
