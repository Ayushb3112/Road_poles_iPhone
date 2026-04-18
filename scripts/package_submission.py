#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


def safe_copy(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    else:
        print(f"Skipped (missing): {src}")


def main() -> None:
    root = Path(".").resolve()
    out_dir = root / "outputs" / "submission_package"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_src = root / "reports" / "metrics_summary.json"
    data_report_src = root / "reports" / "data_check.json"
    preds_dir = root / "outputs" / "test_predictions_labels"

    safe_copy(metrics_src, out_dir / "metrics_summary.json")
    safe_copy(data_report_src, out_dir / "data_check.json")

    if preds_dir.exists():
        preds_zip = out_dir / "test_predictions_labels.zip"
        shutil.make_archive(str(preds_zip.with_suffix("")), "zip", preds_dir)
        print(f"Packed predictions: {preds_zip}")
    else:
        print(f"Skipped (missing): {preds_dir}")

    readme_path = out_dir / "README_submission.txt"
    readme_path.write_text(
        "Submission package contents:\n"
        "- metrics_summary.json (if evaluation was run)\n"
        "- data_check.json\n"
        "- test_predictions_labels.zip (if prediction was run)\n",
        encoding="utf-8",
    )
    print(f"Wrote: {readme_path}")


if __name__ == "__main__":
    main()
