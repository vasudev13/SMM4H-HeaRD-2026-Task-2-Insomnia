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

### 4. Build labeled training JSONL (optional)

Joins `data/training/train_corpus.csv` with `subtask_1.json` / `subtask_2.json` and writes JSONL under `outputs/labeled/`:

```bash
uv run python scripts/build_labeled_datasets.py
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

### 6. Tests

```bash
uv run python -m unittest tests.test_spans -v
```

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
