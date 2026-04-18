#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO


def extract_metrics(metrics) -> Dict[str, Optional[float]]:
    box = metrics.box
    return {
        "precision": float(box.mp) if box.mp is not None else None,
        "recall": float(box.mr) if box.mr is not None else None,
        "mAP50": float(box.map50) if box.map50 is not None else None,
        "mAP50_95": float(box.map) if box.map is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model and export report JSON.")
    parser.add_argument("--data", type=Path, default=Path("configs/data_resolved.yaml"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project", type=str, default="runs/val")
    parser.add_argument("--name", type=str, default="poles_eval")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    val_metrics = model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    report = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "val_metrics": extract_metrics(val_metrics),
        "notes": [
            "Test labels may be hidden for leaderboard datasets.",
            "Use scripts/predict.py outputs for blind test submission if test labels are unavailable.",
        ],
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "metrics_summary.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["val_metrics"], indent=2))
    print(f"Wrote metrics report: {out_file.resolve()}")


if __name__ == "__main__":
    main()
