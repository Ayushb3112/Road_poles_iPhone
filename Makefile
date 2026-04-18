PYTHON ?= python3
DATA_YAML ?= configs/data_resolved.yaml
WEIGHTS ?= runs/train/poles_full_n/weights/best.pt

.PHONY: prepare smoke train_gpu eval predict export package

prepare:
	$(PYTHON) scripts/prepare_dataset.py --root . --allow-missing-test-labels

smoke:
	$(PYTHON) scripts/train.py --data $(DATA_YAML) --model yolov8n.pt --epochs 2 --imgsz 640 --batch 8 --device mps --name poles_smoke

train_gpu:
	$(PYTHON) scripts/train.py --data $(DATA_YAML) --model yolov8n.pt --epochs 100 --imgsz 640 --batch 32 --device 0 --name poles_full_n

eval:
	$(PYTHON) scripts/evaluate.py --data $(DATA_YAML) --weights $(WEIGHTS) --device 0

predict:
	$(PYTHON) scripts/predict.py --weights $(WEIGHTS) --test-list generated/test_abs.txt --device 0

export:
	$(PYTHON) scripts/export_model.py --weights $(WEIGHTS) --format onnx --device 0

package:
	$(PYTHON) scripts/package_submission.py
