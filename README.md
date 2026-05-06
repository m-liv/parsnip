# parsnip

We investigate whether inference-time prompting strategies can mitigate dialect-related performance gaps in LLM reasoning tasks, using the [EnDiVE](https://endiveee.github.io/) benchmark.

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

**Baseline**
```bash
python3 -m src.run_baseline --task wsc --dialect AAVE --model gpt-4o
```

**Dialect-aware**
```bash
python3 -m src.run_dialect_aware --task wsc --dialect AAVE --model gpt-4o
```

**Paraphrase**
```bash
python3 -m src.run_paraphrase --task wsc --dialect AAVE --model gpt-4o
```

**Few-shot**
```bash
python3 -m src.run_few_shot --task wsc --dialect AAVE --model gpt-4o
```

Valid tasks: `wsc`, `multirc`, `logic_bench_mcq`, `folio`  
Valid dialects: `AAVE`, `IndE`  
Valid models: `gemini-2.5-pro`, `gpt-4o`, `gpt-4o-mini`
