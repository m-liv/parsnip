import os
import json
import time
import datetime
from tqdm import tqdm
from itertools import product
from config import TASKS, DIALECTS, MODELS, N, SEED, OUT_DIR, LOG_DIR
from dotenv import load_dotenv
from calls import call_model
from prompt_builder import build_prompt
from data_loader import load_task
from evaluate import parse_task, score
load_dotenv() # Load environment variables from .env (API keys)
from huggingface_hub import login

if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

################################## Output and logging directories ##################################

# Create output and logging directories if they don't exist
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

# Generate path for JSONL output file for a given task, model, dialect, and condition (SAE or dialect)
def jsonl_path(task, model, dialect, condition):
    safe_model = model.replace("/", "_")
    return os.path.join(OUT_DIR, f"{task}__{safe_model}__{dialect}__{condition}.jsonl")

# Generate path for JSONL log file for a given task, model, and dialect
def log_path(task, model, dialect):
    safe_model = model.replace("/", "_")
    return os.path.join(LOG_DIR, f"{task}__{safe_model}__{dialect}.jsonl")

# Append a row (dictionary) to a JSONL file at the given path (creating the file if it doesn't exist)
def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        

################################## Running an experiment ##################################

# Run one task-model-dialect combination: load examples, build prompts, call model, parse and score outputs, and log results
def run_one(task, model, dialect):
    examples = load_task(task, dialect, N, SEED)
    results = {}

    log_file = log_path(task, model, dialect)
    if os.path.exists(log_file):
        os.remove(log_file)

    for condition in ["SAE", dialect]:
        out_path = jsonl_path(task, model, dialect, condition)
        if os.path.exists(out_path):
            os.remove(out_path)

        correct = 0
        total = 0

        for i, ex in enumerate(
            tqdm(
                examples,
                desc=f"{task} | {model} | {dialect} | {condition}",
                leave=False
            )
        ):
            prompt = build_prompt(task, ex, condition)

            start = time.time()
            raw = call_model(prompt, model)
            latency = time.time() - start

            pred = parse_task(task, raw)
            label = ex.get("label")
            is_correct = score(task, pred, label, ex)

            append_jsonl(out_path, {
                "task": task,
                "model": model,
                "dialect": dialect,
                "condition": condition,
                "i": i,
                "prediction": pred,
                "label": label,
                "correct": is_correct,
            })

            append_jsonl(log_file, {
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

        acc = correct / total if total else 0.0
        results[condition] = {"accuracy": acc, "correct": correct, "total": total}

    results["gap"] = results["SAE"]["accuracy"] - results[dialect]["accuracy"]
    return results

def main():
    ensure_dirs()
    summary = []
    
    # For debugging
    d_tasks = ["gsm8k"]
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
                    "sae_accuracy": res["SAE"]["accuracy"],
                    "dialect_accuracy": res[dialect]["accuracy"],
                    "gap": res["gap"],
                })
                print(task, dialect, model, "SAE", res["SAE"]["accuracy"], "DIALECT", res[dialect]["accuracy"], "GAP", res["gap"])

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()