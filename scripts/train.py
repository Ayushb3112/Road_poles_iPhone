#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train snow pole detector with Ultralytics YOLO.")
    parser.add_argument("--data", type=Path, default=Path("configs/data_resolved.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project", type=str, default=str((Path.cwd() / "runs/train").resolve()))
    parser.add_argument("--name", type=str, default="poles")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        pretrained=True,
    )

    run_dir = Path(args.project) / args.name
    print(f"Training complete. Results are in: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
