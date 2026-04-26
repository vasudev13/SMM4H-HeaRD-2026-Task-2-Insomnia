.DEFAULT_GOAL := help

# Test / submission pipeline (defaults mirror data/validation/: data/testing/testing_corpus.csv)
MIMIC_ROOT ?=
TEST_NOTE_IDS ?= data/testing/test_note_ids.txt
TEST_CORPUS ?= data/testing/testing_corpus.csv
INFERENCE_DIR ?= outputs/inference
# Set MAX_ROWS=N to process only the first N notes (smoke test), e.g. make inference-test MAX_ROWS=5
MAX_ROWS ?=
# Nearest-neighbor few-shot count for clean inference mode (0 = zero-shot)
NEIGHBORS ?= 0
GEPA_OUT_DIR ?= outputs/gepa/optimized
GEPA_BASELINE_OUT_DIR ?= outputs/gepa/baseline
GEPA_MAX_METRIC_CALLS ?= 150
GEPA_TRAIN_LIMIT ?= 30

.PHONY: help sync baml-init baml-generate build-labeled inference inference-test \
	inference-clean gepa test-corpus example-index sanitize-subtask2 submission test clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install and sync dependencies with uv
	uv sync

baml-init: ## Create baml_src/insomnia.baml from example if missing
	@if [ -f baml_src/insomnia.baml ]; then \
		echo "baml_src/insomnia.baml already exists"; \
	else \
		cp baml_src/insomnia.baml.example baml_src/insomnia.baml; \
		echo "Created baml_src/insomnia.baml"; \
	fi

baml-generate: ## Generate BAML Python client
	uv run baml-cli generate

build-labeled: ## Build labeled training JSONL outputs
	uv run python scripts/build_labeled_datasets.py

inference: ## Run insomnia inference on validation corpus
	uv run insomnia-inference

test-corpus: ## Build TEST_CORPUS from TEST_NOTE_IDS + MIMIC_ROOT (set MIMIC_ROOT)
	@test -n "$(MIMIC_ROOT)" || (echo "Set MIMIC_ROOT to your MIMIC-III v1.4 CSV directory (NOTEEVENTS.csv.gz, etc.)." && exit 1)
	@test -f "$(TEST_NOTE_IDS)" || (echo "Missing $(TEST_NOTE_IDS)" && exit 1)
	@mkdir -p "$(dir $(TEST_CORPUS))"
	uv run python text_mimic_notes.py --note_ids_path "$(TEST_NOTE_IDS)" --mimic_path "$(MIMIC_ROOT)" --output_path "$(TEST_CORPUS)"

inference-test: ## Run inference on TEST_CORPUS → INFERENCE_DIR (v2, few-shot, no throttle; optional MAX_ROWS)
	@test -f "$(TEST_CORPUS)" || (echo "Missing $(TEST_CORPUS); run 'make test-corpus MIMIC_ROOT=...' first." && exit 1)
	uv run insomnia-inference --input-csv "$(TEST_CORPUS)" --out-dir "$(INFERENCE_DIR)" \
		--use-few-shot --min-interval-sec 0 \
		$(if $(strip $(MAX_ROWS)),--max-rows $(MAX_ROWS),)

inference-clean: ## Run inference-test with configurable NEIGHBORS (default 0 = zero-shot)
	@test -f "$(TEST_CORPUS)" || (echo "Missing $(TEST_CORPUS); run 'make test-corpus MIMIC_ROOT=...' first." && exit 1)
	uv run insomnia-inference --input-csv "$(TEST_CORPUS)" --out-dir "$(INFERENCE_DIR)" \
		$(if $(filter 0,$(NEIGHBORS)),--no-few-shot,--use-few-shot --few-shot-k $(NEIGHBORS)) \
		--min-interval-sec 0 \
		$(if $(strip $(MAX_ROWS)),--max-rows $(MAX_ROWS),)

gepa: ## Run GEPA pipeline (baseline + optimize + compare), configurable via NEIGHBORS/MAX_ROWS
	uv run python scripts/run_gepa.py \
		--baseline-neighbors "$(NEIGHBORS)" \
		$(if $(strip $(MAX_ROWS)),--baseline-max-rows $(MAX_ROWS),) \
		--max-metric-calls "$(GEPA_MAX_METRIC_CALLS)" \
		--train-limit "$(GEPA_TRAIN_LIMIT)" \
		--baseline-out-dir "$(GEPA_BASELINE_OUT_DIR)" \
		--gepa-out-dir "$(GEPA_OUT_DIR)"

example-index: ## Build FAISS few-shot index (data/training → data/)
	uv run python scripts/build_example_index.py

sanitize-subtask2: ## Sanitize subtask_2.json in INFERENCE_DIR in place
	@test -f "$(INFERENCE_DIR)/subtask_2.json" || (echo "Missing $(INFERENCE_DIR)/subtask_2.json; run inference first." && exit 1)
	uv run python scripts/sanitize_subtask2_predictions.py "$(INFERENCE_DIR)/subtask_2.json" --in-place

submission: ## Create date-stamped ZIPs from INFERENCE_DIR JSONs
	@test -f "$(INFERENCE_DIR)/subtask_1.json" || (echo "Missing $(INFERENCE_DIR)/subtask_1.json; run inference (e.g. make inference-test) first." && exit 1)
	@test -f "$(INFERENCE_DIR)/subtask_2.json" || (echo "Missing $(INFERENCE_DIR)/subtask_2.json; run inference (e.g. make inference-test) first." && exit 1)
	@d=$$(date +%Y%m%d); \
	rm -f "$(INFERENCE_DIR)/subtask_1_$$d.zip" "$(INFERENCE_DIR)/subtask_2_$$d.zip"; \
	zip -j "$(INFERENCE_DIR)/subtask_1_$$d.zip" "$(INFERENCE_DIR)/subtask_1.json" >/dev/null; \
	zip -j "$(INFERENCE_DIR)/subtask_2_$$d.zip" "$(INFERENCE_DIR)/subtask_2.json" >/dev/null; \
	echo "Created $(INFERENCE_DIR)/subtask_1_$$d.zip"; \
	echo "Created $(INFERENCE_DIR)/subtask_2_$$d.zip"

test: ## Run unit tests
	uv run python -m unittest tests.test_spans -v

clean: ## Remove subtask JSONs from INFERENCE_DIR
	rm -f "$(INFERENCE_DIR)/subtask_1.json" "$(INFERENCE_DIR)/subtask_2.json"
