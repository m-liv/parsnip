# ParSnip

We investigate whether inference-time prompting strategies can mitigate dialect-related performance gaps in LLM reasoning tasks, using the [EnDive](https://endiveee.github.io/) benchmark.

## Key Findings
- **WSC shows the largest dialect gaps**, scaling with model capability on IndE (up to 30 pp for Gemini 2.5 Pro)
- **Few-shot prompting** is the most robust intervention, achieving full gap closure in multiple conditions
- **Paraphrasing** helps on reading comprehension tasks but severely harms tasks sensitive to precise wording (e.g., −18.5 pp on WSC IndE for GPT-4o)
- **Dialect-aware framing** is model-dependent: beneficial for Gemini 2.5 Pro, harmful for GPT-4o-mini on several tasks

## Research Overview

We evaluate models on dialectal variants of four reasoning tasks, comparing performance on Standard American English (SAE) inputs vs. dialectal inputs across two non-standard dialects:

- **AAVE** — African American Vernacular English
- **IndE** — Indian English

**Tasks:** WSC (coreference), MultiRC (reading comprehension), Logic Bench MCQ (logical reasoning), FOLIO (formal logic)

**Models:** Gemini 2.5 Pro, GPT-4o, GPT-4o mini

**Prompting strategies:**

| Strategy | Description |
|---|---|
| Baseline | No intervention; prompt in SAE or dialect as-is |
| Dialect-aware | Prompt includes a statement identifying the dialect |
| Few-shot | Prompt includes 3 dialectal in-context examples |
| Paraphrase | Dialect text is first paraphrased to SAE, then the task prompt runs on the paraphrase |

Results and logs are written to `results/<strategy>/` and `logs/<strategy>/`.

## Output Format
Each result file is a JSON log containing per-instance predictions, ground truth labels, and accuracy. Summary metrics (task accuracy, Δ pp, and gap closure %) are printed at the end of each run and written to `results/<strategy>/<task>_<dialect>_<model>.json`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env-example` to `.env` and fill in your API keys:

```
OPENAI_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
OPENROUTER_API_KEY=
```

## Running Experiments

Each pipeline script accepts `--task`, `--dialect`, and `--model` arguments. Valid values are defined in [src/config.py](src/config.py).

To run from the `src` directory:

**Baseline**
```bash
python3 -m run_baseline.py --task wsc --dialect AAVE --model gpt-4o
```

**Dialect-aware**
```bash
python3 -m run_dialect_aware.py --task wsc --dialect AAVE --model gpt-4o
```

**Paraphrase**
```bash
python3 -m run_paraphrase.py --task wsc --dialect AAVE --model gpt-4o
```

**Few-shot**
```bash
python3 -m run_few_shot.py --task wsc --dialect AAVE --model gpt-4o
```

Valid tasks: `wsc`, `multirc`, `logic_bench_mcq`, `folio`  
Valid dialects: `AAVE`, `IndE`  
Valid models: `gemini-2.5-pro`, `gpt-4o`, `gpt-4o-mini`

## Acknowledgments
This project builds on the [EnDive benchmark](https://endiveee.github.io/) by Gupta et al. (2025). Dialectal task data was sourced from EnDive; to access the dataset, follow the instructions on the EnDive project page.
