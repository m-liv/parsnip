import os
import json
import time
import datetime
from tqdm import tqdm
from config import N, SEED
from calls import call_model
from prompt_builder import build_prompt, build_few_shot_prompt
from data_loader import load_task
from evaluate import parse_task, score
from huggingface_hub import login

if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

OUT_DIR = "results/few_shot"
LOG_DIR = "logs/few_shot"

################################## Output and logging directories ##################################

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def json_path(task, model, dialect, condition):
    safe_model = model.replace("/", "_")
    return os.path.join(OUT_DIR, f"FS__{task}__{safe_model}__{dialect}__{condition}.json")

def log_path(task, model, dialect):
    safe_model = model.replace("/", "_")
    return os.path.join(LOG_DIR, f"FS__{task}__{safe_model}__{dialect}.json")

def write_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

################################## Few-shot deduplication ##################################

# Maps each task to the key in the example dict and the key in the shot dict
# that hold the primary dialectal text, used to detect overlap.
_DIALECT_FIELD = {
    "wsc":             ("dialect_paragraph", "paragraph"),
    "logic_bench_mcq": ("dialect_input",     "context"),
    "multirc":         ("dialect_input",     "paragraph"),
    "boolq":           ("dialect_passage",   "passage"),
    "gsm8k":           ("dialect_question",  "question"),
    "mbpp":            ("dialect_prompt",    "problem"),
    "folio":           ("dialect_input",     "premises"),
}

def get_skip_indices(task, examples, dialect):
    """Return the set of example indices whose dialect text matches a few-shot shot."""
    from few_shot_examples import FEW_SHOT_EXAMPLES
    shots = FEW_SHOT_EXAMPLES.get(task, {}).get(dialect, [])
    if not shots:
        return set()
    ex_field, shot_field = _DIALECT_FIELD.get(task, (None, None))
    if ex_field is None:
        return set()
    shot_texts = {s[shot_field] for s in shots}
    return {i for i, ex in enumerate(examples) if ex.get(ex_field) in shot_texts}

################################## Running an experiment ##################################

def run_one(task, model, dialect):
    examples = load_task(task, dialect, N, SEED)
    skip = get_skip_indices(task, examples, dialect)
    if skip:
        print(f"Skipping {len(skip)} example(s) that appear in the few-shot pool: indices {sorted(skip)}")
    results = {}

    log_file = log_path(task, model, dialect)
    if os.path.exists(log_file):
        os.remove(log_file)

    log_rows = []

    for condition in [dialect]:
        out_path = json_path(task, model, dialect, condition)
        if os.path.exists(out_path):
            os.remove(out_path)

        results_rows = []
        correct = 0
        total = 0

        for i, ex in enumerate(
            tqdm(
                examples,
                desc=f"{task} | {model} | {dialect} | {condition} | FS",
                leave=False,
            )
        ):
            if i in skip:
                continue

            # SAE condition uses the standard zero-shot prompt; dialect condition uses few-shot
            if condition == "SAE":
                prompt = build_prompt(task, ex, "SAE")
            else:
                prompt = build_few_shot_prompt(task, ex, dialect)

            start = time.time()
            raw = call_model(prompt, model)
            latency = time.time() - start

            pred = parse_task(task, raw)
            label = ex.get("label")
            is_correct = score(task, pred, label, ex)

            results_rows.append({
                "task": task,
                "model": model,
                "dialect": dialect,
                "condition": condition,
                "i": i,
                "prediction": pred,
                "label": label,
                "correct": is_correct,
            })

            log_rows.append({
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "task": task,
                "model": model,
                "dialect": dialect,
                "condition": condition,
                "example_index": i,
                "latency_seconds": latency,
                "prompt": prompt,
                "response": raw,
            })

            correct += 1 if is_correct else 0
            total += 1

            time.sleep(0.05)

        write_json(out_path, results_rows)

        acc = correct / total if total else 0.0
        results[condition] = {"accuracy": acc, "correct": correct, "total": total}

    write_json(log_file, log_rows)

    return results

def main():
    ensure_dirs()
    summary = []

    # For debugging
    d_tasks = ["wsc"]
    d_dialects = ["AAVE"]
    d_models = ["gpt-4o"]

    for task in d_tasks:
        for dialect in d_dialects:
            for model in d_models:
                res = run_one(task, model, dialect)
                summary.append({
                    "task": task,
                    "dialect": dialect,
                    "model": model,
                    "dialect_accuracy": res[dialect]["accuracy"],
                })
                print(task, dialect, model, "DIALECT", res[dialect]["accuracy"])

    with open(os.path.join(OUT_DIR, "summary_few_shot.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
