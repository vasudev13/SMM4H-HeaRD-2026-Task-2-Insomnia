# SMM4H-HeaRD @ ACL-2026 Shared Task 2: Detection of Insomnia in Clinical Notes

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv, then create a virtual environment and install dependencies:

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync
```

Alternatively, run scripts without activating the environment:

```bash
uv run python text_mimic_notes.py ...
```

The project targets **Python 3.10+** (see `pyproject.toml`).

## Running this repository

### 1. Install dependencies

From the repository root:

```bash
uv sync
```

This installs the package (including the `insomnia` library), BAML, pandas, and other dependencies.

### 2. API key (LLM inference)

Inference uses **Google Gemini** via BAML (`CustomGemini` in `baml_src/clients.baml`). Create a `.env` file in the repo root (you can start from `.env.example`) and set:

```bash
GOOGLE_API_KEY=your_key_here
```

The inference CLI loads `.env` automatically before calling the model.

`CustomGemini` in `baml_src/clients.baml` is tuned for **structured extraction**: `generationConfig` sets **temperature 0** so repeated runs on the same note stay consistent, and **max output tokens 8192** so long clinical notes with many criteria are less likely to hit **silent truncation** (which breaks BAML JSON parsing). **Gemini 2.5 Pro extended thinking** is requested via `thinkingConfig` nested under `generationConfig` in `clients.baml` (dynamic thinking budget); adjust or remove that block if you switch models. Placing `thinkingConfig` at the top level of the request is rejected by the Google AI API (400).

### 3. BAML prompts and generated client

**`baml_src/insomnia.baml` is not committed** (private prompts; see `.gitignore`). The repo includes **[`baml_src/insomnia.baml.example`](baml_src/insomnia.baml.example)** with the same types and placeholder prompts.

On a fresh clone (or first run):

1. Copy the example to the real filename and edit prompts (task rules are summarized in [`resources/Insomnia_Rules.md`](resources/Insomnia_Rules.md)):

   ```bash
   cp baml_src/insomnia.baml.example baml_src/insomnia.baml
   ```

2. Generate the Python client (creates **`baml_client/`** locally; that directory is also gitignored because `inlinedbaml.py` embeds BAML sources):

   ```bash
   uv run baml-cli generate
   ```

Re-run `baml-cli generate` whenever you change any file under `baml_src/`.

### 4. Build labeled train/validation artifacts (optional)

Builds labeled artifacts for `training` and `validation` (default) by joining each split corpus CSV with its `subtask_1.json` and `subtask_2.json`.

Default run:

```bash
uv run python scripts/build_labeled_datasets.py
```

This writes split-scoped outputs under `outputs/labeled/`:

- `outputs/labeled/training/subtask1_labeled.jsonl`
- `outputs/labeled/training/subtask2_labeled.jsonl`
- `outputs/labeled/training/subtask2_labeled_text_only.jsonl`
- `outputs/labeled/training/training_labeled.csv`
- `outputs/labeled/validation/subtask1_labeled.jsonl`
- `outputs/labeled/validation/subtask2_labeled.jsonl`
- `outputs/labeled/validation/subtask2_labeled_text_only.jsonl`
- `outputs/labeled/validation/validation_labeled.csv`

CSV schema (one row per note):

- `note_id`
- `text`
- `subtask1_label` (`yes`/`no`)
- `def1_spans`
- `def2_spans`
- `ruleb_spans`
- `rulec_spans`

Each `*_spans` column is stored as a JSON array string (e.g. `["881 899"]`).

Useful flags:

```bash
# Build one split only
uv run python scripts/build_labeled_datasets.py --split training
uv run python scripts/build_labeled_datasets.py --split validation

# Override paths for a single split
uv run python scripts/build_labeled_datasets.py \
  --split training \
  --corpus path/to/corpus.csv \
  --subtask1 path/to/subtask_1.json \
  --subtask2 path/to/subtask_2.json
```

### 5. Run inference (validation → submission-shaped JSON)

Default input is `data/validation/validation_corpus.csv`; default output directory is `outputs/inference/` (`subtask_1.json`, `subtask_2.json`):

```bash
uv run insomnia-inference
```

Equivalent:

```bash
uv run python scripts/run_inference.py
```

Useful flags:

```bash
uv run insomnia-inference --input-csv path/to/notes.csv --out-dir path/to/out --max-rows 5
```

`--max-rows` limits how many notes are processed (handy for a quick smoke test).

**Gemini rate limits:** Inference issues one model call per note (`ExtractClinicalEvidence`). On the **Google AI free tier**, `gemini-2.5-flash` is often capped at about **5 requests per minute**, which leads to HTTP **429** if calls are fired back-to-back. By default, the CLI waits **12 seconds** between the *start* of consecutive requests (~5/min). Override with `GEMINI_MIN_INTERVAL_SEC` in `.env`, or `--min-interval-sec SEC` / `--rpm N` (mutually exclusive alternatives, where `--rpm` sets an interval of `60/N` seconds). Use `--min-interval-sec 0` when your API quota is high enough that throttling is unnecessary.

### 6. Tests

```bash
uv run python -m unittest tests.test_spans tests.test_evaluate -v
```

### 7. Evaluate predictions (Task 2 metrics)

This repository now includes a local evaluator for:

- Subtask 1: F1 score (`yes` as positive class)
- Subtask 2A: micro-average F1 over rule labels
- Subtask 2B: macro-average ROUGE-L Precision/Recall/F1 over evidence text

Run with defaults (gold from `data/training/`, predictions from `outputs/inference/`):

```bash
uv run insomnia-evaluate --split-name train
```

Equivalent:

```bash
uv run python scripts/evaluate_predictions.py --split-name train
```

Explicit paths example:

```bash
uv run insomnia-evaluate \
  --gold-subtask1 data/training/subtask_1.json \
  --gold-subtask2 data/training/subtask_2.json \
  --pred-subtask1 outputs/inference/subtask_1.json \
  --pred-subtask2 outputs/inference/subtask_2.json \
  --split-name validation \
  --json-out outputs/inference/metrics.json
```

With `--json-out`, metrics are written under `--experiment-name` (default `v0 - gemini-2.5-flash baseline`), with `split_name` included in that nested object. Use `--experiment-name ''` for the previous flat shape.

Notes:

- Official Task 2 scorer code is not currently bundled in this repository.
- The local evaluator is intended for reproducible offline model iteration and should be treated as an approximation until organizers release an official scorer endpoint/script.
- Subtask 2B local scoring averages ROUGE-L over `Definition 1`, `Definition 2`, `Rule B`, and `Rule C` entries where the gold label is `yes`.

## Makefile shortcuts

For convenience, you can use the repository `Makefile`:

```bash
make sync
make baml-init
make baml-generate
make inference
make test
```

Run `make` (or `make help`) to see all available targets.

### Test split (submission E2E)

The test split mirrors the validation layout: note IDs and corpus CSV live under **`data/testing/`** — by default **`data/testing/test_note_ids.txt`** (one integer `ROW_ID` per line) and the MIMIC-built corpus **`data/testing/testing_corpus.csv`** (same role as `data/validation/validation_corpus.csv`). Inference still writes **`outputs/inference/subtask_1.json`** and **`subtask_2.json`** unless you override **`INFERENCE_DIR`**.

```bash
make test-corpus MIMIC_ROOT=/path/to/mimic-iii/1.4
make inference-test
make sanitize-subtask2   # optional
make submission
```

Override paths with Makefile variables, e.g. `TEST_NOTE_IDS=...`, `TEST_CORPUS=...`, `INFERENCE_DIR=...`. Use `MAX_ROWS=N` on `inference-test` for a short smoke run.

## Corpus

This shared task utilizes a corpus of clinical notes derived from the MIMIC-III Database. The clinical notes have been augmented with additional structured patient information, specifically sex, age, and the medications prescribed during their hospital stay.

Participants are required to complete necessary training and sign a data usage agreement to access the [MIMIC-III Clinical Database (v1.4)](https://physionet.org/content/mimiciii/1.4/). After gaining access and downloading the files, participants must run the [`text_mimic_notes.py`](text_mimic_notes.py) script to retrieve clinical notes and associated patient information using the provided note IDs. This process builds the corpus utilized in this shared task, as detailed in the instructions provided below.

### MIMIC-III Notes Processing

The `text_mimic_notes.py` Python script is designed to retrieve clinical notes and patient information from the MIMIC-III clinical database. The script takes a text file containing note IDs, and merges it with the content of the notes from MIMIC-III, including additional demographic and prescription information.

#### Requirements

- Python 3.10 or higher (project default)
- pandas library
- datetime module

#### Usage

The script requires three command-line arguments:
- `--note_ids_path`: The file path to the text file containing the note IDs.
- `--mimic_path`: The directory path containing the MIMIC-III v1.4 CSV files (`NOTEEVENTS.csv.gz`, `PRESCRIPTIONS.csv.gz` and `PATIENTS.csv.gz`).
- `--output_path`: The file path where the processed corpus CSV will be saved. This output CSV file will have two columns: the note IDs and the textual data retrieved from MIMIC-III.

#### Command Syntax

The script is executed from the command line with the following syntax:

```bash
python text_mimic_notes.py --note_ids_path [path_to_note_ids_txt] --mimic_path [path_to_mimic_csv_directory] --output_path [path_to_output_csv]
```

#### Example Command

Here is an example command that illustrates how to run the script using specific paths for each required input:

```bash
python text_mimic_notes.py --note_ids_path ./training/sample_note_ids.txt  --mimic_path ./mimic-iii/1.4 --output_path ./training/sample_corpus.csv
```

This command will process the note IDs from `./training/sample_note_ids.txt`, merge them with the data found in `./mimic-iii/1.4`, and output the resulting corpus to `./training/sample_corpus.csv`.
