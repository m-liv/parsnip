import os
import json
import time
import datetime
from tqdm import tqdm
from itertools import product
from config import TASKS, DIALECTS, MODELS, N, SEED, OUT_DIR, LOG_DIR
from calls import call_model
from prompt_builder import build_prompt, build_dialect_aware_prompt
from data_loader import load_task
from evaluate import parse_task, score
from huggingface_hub import login

if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

################################## Output and logging directories ##################################

# Create output and logging directories if they don't exist
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

# Generate path for JSON output file for a given task, model, dialect, and condition (SAE or dialect)
def json_path(task, model, dialect, condition):
    safe_model = model.replace("/", "_")
    return os.path.join(OUT_DIR, f"DA__{task}__{safe_model}__{dialect}__{condition}.json")

# Generate path for JSON log file for a given task, model, and dialect
def log_path(task, model, dialect):
    safe_model = model.replace("/", "_")
    return os.path.join(LOG_DIR, f"DA__{task}__{safe_model}__{dialect}.json")

# Write a list of rows (dictionaries) to a JSON file at the given path
def write_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

################################## Running an experiment ##################################

# Run one task-model-dialect combination: load examples, build prompts, call model, parse and score outputs, and log results
def run_one(task, model, dialect):
    examples = load_task(task, dialect, N, SEED)
    results = {}

    log_file = log_path(task, model, dialect)
    if os.path.exists(log_file):
        os.remove(log_file)

    log_rows = []
    
    for condition in ["SAE", dialect]:
        out_path = json_path(task, model, dialect, condition)
        if os.path.exists(out_path):
            os.remove(out_path)

        results_rows = []
        
        correct = 0
        total = 0

        for i, ex in enumerate(
            tqdm(
                examples,
                desc=f"{task} | {model} | {dialect} | {condition}",
                leave=False
            )
        ):
            prompt = build_dialect_aware_prompt(task, ex, condition, dialect)

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
                "timestamp": datetime.datetime.utcnow().isoformat(),
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
    
    results["gap"] = results["SAE"]["accuracy"] - results[dialect]["accuracy"]
    return results

def main():
    ensure_dirs()
    summary = []
    
    # For debugging
    d_tasks = ["wsc"]
    d_dialects = ["IndE"]
    d_models = ["gpt-4o"]
    
    for task in d_tasks:
        for dialect in d_dialects:
            for model in d_models:
                res = run_one(task, model, dialect)
                summary.append({
                    "task": task,
                    "dialect": dialect,
                    "model": model,
                    "sae_accuracy": res["SAE"]["accuracy"],
                    "dialect_accuracy": res[dialect]["accuracy"],
                    "gap": res["gap"],
                })
                print(task, dialect, model, "SAE", res["SAE"]["accuracy"], "DIALECT", res[dialect]["accuracy"], "GAP", res["gap"])

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()