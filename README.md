# SMM4H-HeaRD @ ACL 2026 -- Shared Task 2: Detection of Insomnia in Clinical Notes

A two-pass pipeline for detecting insomnia in MIMIC-III clinical notes. The system uses **Google Gemini** (via [BAML](https://docs.boundaryml.com/)) for structured evidence extraction and deterministic rules for label derivation, with optional **GEPA** (Genetic Prompt Algorithm) prompt optimization.

## Architecture

```
Clinical Note
     │
     ▼
┌─────────────────────────────┐
│  Pass 1: LLM Extraction     │  Gemini 2.5 Flash / Pro
│  (baml_src/insomnia.baml)    │  → structured JSON evidence
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Pass 2: Rule Engine         │  deterministic label derivation
│  (insomnia/inference.py)     │  → Definition 1, Definition 2,
│                              │    Rule A, Rule B, Rule C
└──────────────┬──────────────┘
               │
               ▼
  subtask_1.json + subtask_2.json
```

**Pass 1** extracts sleep difficulties, daytime impairments, and medication mentions as structured evidence with verbatim citations. **Pass 2** applies the task's clinical rules deterministically to produce labels and character-level spans.

## Quick Start

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone git@github.com:vasudev13/SMM4H-HeaRD-2026-Task-2-Insomnia.git
cd SMM4H-HeaRD-2026-Task-2-Insomnia
uv sync

# 3. Set up API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 4. Set up BAML prompts and generate client
cp baml_src/insomnia.baml.example baml_src/insomnia.baml
# Edit baml_src/insomnia.baml with your prompt instructions
uv run baml-cli generate

# 5. Build the corpus from MIMIC-III
make test-corpus MIMIC_ROOT=/path/to/mimic-iii/1.4

# 6. Run inference
make inference-test
```

## Project Structure

```
├── baml_src/                    # BAML prompt definitions
│   ├── insomnia.baml            # Main extraction prompt (two-pass v2)
│   ├── insomnia.baml.example    # Template with placeholder prompts
│   ├── clients.baml             # LLM client configuration (Gemini)
│   └── generators.baml          # BAML code-gen settings
├── insomnia/                    # Core Python package
│   ├── inference.py             # Two-pass inference pipeline + derive_labels()
│   ├── evaluate.py              # Local evaluator (F1, ROUGE-L)
│   ├── evaluate_cli.py          # CLI wrapper for evaluation
│   ├── spans.py                 # Character-span utilities
│   ├── gepa_optimize.py         # GEPA prompt optimization module
│   ├── retrieve_few_shots.py    # FAISS-based few-shot retrieval
│   └── format_few_shots.py      # Few-shot example formatting
├── scripts/                     # Standalone utility scripts
│   ├── run_inference.py         # Thin wrapper → insomnia-inference
│   ├── run_gepa.py              # GEPA optimization runner
│   ├── build_labeled_datasets.py # Build labeled train/val artifacts
│   ├── build_example_index.py   # Build FAISS few-shot index
│   ├── evaluate_predictions.py  # Thin wrapper → insomnia-evaluate
│   └── sanitize_subtask2_predictions.py  # Fix span formatting
├── tests/                       # Unit tests
├── resources/                   # Reference documents & annotation guidelines
├── text_mimic_notes.py          # MIMIC-III corpus builder
├── Makefile                     # Task automation
├── pyproject.toml               # Project metadata & dependencies
└── .env.example                 # API key template
```

## Requirements

- **Python 3.10+**
- **uv** for dependency management
- **MIMIC-III v1.4** access ([PhysioNet](https://physionet.org/content/mimiciii/1.4/))
- **Google AI API key** (Gemini 2.5 Flash or Pro)

## Configuration

### API Key

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Set `GOOGLE_API_KEY=your_key_here`. The inference CLI loads this automatically.

### BAML Prompts

`baml_src/insomnia.baml` contains the extraction prompt. On a fresh clone, start from the example:

```bash
cp baml_src/insomnia.baml.example baml_src/insomnia.baml
```

See [`resources/Insomnia_Rules.md`](resources/Insomnia_Rules.md) for the clinical definitions (Definition 1, Definition 2, Rules A/B/C). After editing, regenerate the Python client:

```bash
uv run baml-cli generate
```

### Gemini Settings

`baml_src/clients.baml` configures `CustomGemini` with **temperature 0** for reproducibility and **8192 max output tokens** to avoid silent truncation on long notes. Extended thinking is enabled via `thinkingConfig` in `generationConfig`.

## Usage

### Building the Corpus

The shared task requires MIMIC-III access. After downloading, build the corpus:

```bash
make test-corpus MIMIC_ROOT=/path/to/mimic-iii/1.4
```

This runs `text_mimic_notes.py` to join note IDs with MIMIC-III note text, demographics, and prescriptions.

### Running Inference

```bash
# Full test set (with few-shot examples, no API throttle)
make inference-test

# Zero-shot mode
make inference-clean NEIGHBORS=0

# Few-shot mode with k neighbors
make inference-clean NEIGHBORS=3

# Smoke test (first 5 notes)
make inference-test MAX_ROWS=5
```

Output is written to `outputs/inference/subtask_1.json` and `subtask_2.json`.

**Rate limiting:** On the Google AI free tier, Gemini may cap at ~5 RPM. The default 12-second interval handles this. Override with `--min-interval-sec 0` when your quota allows it (all `make` targets above disable throttling).

### GEPA Prompt Optimization

[GEPA](https://github.com/GeneticsOfPrompting/gepa) evolves the extraction prompt using genetic algorithms while keeping the deterministic rule engine fixed:

```bash
make gepa
```

Configure with `GEPA_MAX_METRIC_CALLS`, `GEPA_TRAIN_LIMIT`, and `NEIGHBORS`.

### Evaluation

Local evaluation against gold labels (approximates the official scorer):

```bash
# Evaluate against training gold
uv run insomnia-evaluate --split-name train

# Evaluate against validation gold
uv run insomnia-evaluate --split-name validation

# Custom paths
uv run insomnia-evaluate \
  --gold-subtask1 data/validation/subtask_1.json \
  --gold-subtask2 data/validation/subtask_2.json \
  --pred-subtask1 outputs/inference/subtask_1.json \
  --pred-subtask2 outputs/inference/subtask_2.json
```

Metrics: Subtask 1 F1, Subtask 2A micro-F1 (rule labels), Subtask 2B macro ROUGE-L (evidence spans).

### Creating Submissions

```bash
make sanitize-subtask2   # optional: fix span formatting
make submission          # creates date-stamped ZIPs
```

### Tests

```bash
make test
```

## Makefile Reference

| Target | Description |
|---|---|
| `sync` | Install/sync dependencies |
| `baml-init` | Create `insomnia.baml` from example |
| `baml-generate` | Regenerate BAML Python client |
| `build-labeled` | Build labeled training/validation JSONL |
| `inference` | Run inference on validation corpus |
| `inference-test` | Run inference on test corpus (few-shot) |
| `inference-clean` | Run inference with configurable `NEIGHBORS` |
| `gepa` | Run GEPA prompt optimization pipeline |
| `test-corpus` | Build test corpus from MIMIC-III |
| `example-index` | Build FAISS few-shot index |
| `sanitize-subtask2` | Fix span formatting in subtask_2.json |
| `submission` | Create date-stamped submission ZIPs |
| `test` | Run unit tests |
| `clean` | Remove inference outputs |

## Data Layout

```
data/
├── training/           # Gold labels + note IDs (provided by organizers)
├── validation/         # Gold labels + note IDs (provided by organizers)
├── testing/            # Test note IDs + built corpus
├── example_metadata.json  # Few-shot example metadata
└── faiss_index.bin     # Pre-built FAISS index for retrieval
```

All files under `data/` are gitignored due to MIMIC-III data use agreements.

## License

This project is part of the [SMM4H-HeaRD 2026 Shared Task](https://healthlanguageprocessing.org/smm4h-2026/). The clinical notes are derived from MIMIC-III and subject to its [data use agreement](https://physionet.org/content/mimiciii/1.4/).
