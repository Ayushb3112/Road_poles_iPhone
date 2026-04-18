# Snow Pole Detection (Road Poles 2025)

This project provides a reproducible YOLO pipeline for:
- dataset validation and path normalization,
- training on Mac (smoke test) and NVIDIA GPU (full run),
- evaluation with assignment metrics: Precision, Recall, mAP@50, mAP@50-95,
- generating blind-test predictions for submission.

## 1) Environment setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Prepare dataset files

Your split files currently point to `data/images/...`; this script normalizes them to your local folder structure and validates labels.

```bash
python scripts/prepare_dataset.py --root . --allow-missing-test-labels
```

Outputs:
- `configs/data_resolved.yaml`
- `generated/train_abs.txt`, `generated/val_abs.txt`, `generated/test_abs.txt`
- `reports/data_check.json`

## 3) Mac smoke training (sanity check)

```bash
python scripts/train.py \
  --data configs/data_resolved.yaml \
  --model yolov8n.pt \
  --epochs 2 \
  --imgsz 640 \
  --batch 8 \
  --device mps \
  --name poles_smoke
```

If `mps` has issues, use `--device cpu`.

## 4) Full training on NVIDIA machine

Copy this whole project folder to your NVIDIA machine, then run:

```bash
python scripts/train.py \
  --data configs/data_resolved.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --name poles_full_n
```

Optional stronger baseline:

```bash
python scripts/train.py \
  --data configs/data_resolved.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 24 \
  --device 0 \
  --name poles_full_s
```

## 5) Evaluation (required metrics)

```bash
python scripts/evaluate.py \
  --data configs/data_resolved.yaml \
  --weights runs/train/poles_full_n/weights/best.pt \
  --device 0
```

This writes `reports/metrics_summary.json` with:
- `precision`
- `recall`
- `mAP50`
- `mAP50_95`

## 6) Generate test predictions (leaderboard/blind set)

```bash
python scripts/predict.py \
  --weights runs/train/poles_full_n/weights/best.pt \
  --test-list generated/test_abs.txt \
  --device 0
```

Prediction text files are written to `outputs/test_predictions_labels`.

## 7) Export model (optional for deployment)

```bash
python scripts/export_model.py \
  --weights runs/train/poles_full_n/weights/best.pt \
  --format onnx \
  --device 0
```

## Notes

- Single class is fixed as `pole` with class ID `0`.
- Test labels may be hidden by design; evaluation metrics are typically computed on validation, while test is used for submission scoring.
- `Makefile` shortcuts are available: `make prepare`, `make smoke`, `make train_gpu`, `make eval`, `make predict`, `make export`, `make package`.
