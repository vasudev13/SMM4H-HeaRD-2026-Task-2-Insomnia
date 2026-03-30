.DEFAULT_GOAL := help

.PHONY: help sync baml-init baml-generate build-labeled inference test clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

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

test: ## Run unit tests
	uv run python -m unittest tests.test_spans -v

clean: ## Remove generated inference outputs
	rm -f outputs/inference/subtask_1.json outputs/inference/subtask_2.json
