PYTHON ?= python
RAW_IBES ?= admin/local/wrds-downloads/tr_ibes_11289435.csv
PROCESSED_DIR ?= data/processed/ibes_lora_baseline

.PHONY: doctor check-data summarize-ibes prepare-ibes-small prepare-ibes train-smoke

doctor:
	$(PYTHON) scripts/doctor.py

check-data:
	$(PYTHON) scripts/check_wrds_data.py --input $(RAW_IBES)

summarize-ibes:
	$(PYTHON) scripts/summarize_ibes.py --input $(RAW_IBES)

prepare-ibes-small:
	$(PYTHON) scripts/prepare_ibes_dataset.py --input $(RAW_IBES) --out $(PROCESSED_DIR) --skip-10k

prepare-ibes:
	$(PYTHON) scripts/prepare_ibes_dataset.py --input $(RAW_IBES) --out $(PROCESSED_DIR)

train-smoke:
	$(PYTHON) training/train_smoke.py --model-id Qwen/Qwen3.6-27B --max-steps 1 --max-seq-length 1024 --local-files-only
