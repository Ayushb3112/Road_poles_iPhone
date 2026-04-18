#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained YOLO model (e.g., ONNX).")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", type=str, default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    output_path = model.export(format=args.format, imgsz=args.imgsz, device=args.device)
    print(f"Export complete: {output_path}")


if __name__ == "__main__":
    main()
