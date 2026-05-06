import os
import json
import time
import datetime
from tqdm import tqdm
from config import TASKS, DIALECTS, MODELS, N, SEED
from calls import call_model
from prompt_builder import get_dialect_text, build_paraphrase_prompt, build_prompt_from_paraphrase
from data_loader import load_task
from evaluate import parse_task, score
from huggingface_hub import login

if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
    
OUT_DIR = "results/paraphrase"
LOG_DIR = "logs/paraphrase"

################################## Output and logging directories ##################################

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def json_path(task, model, dialect, condition):
    safe_model = model.replace("/", "_")
    return os.path.join(OUT_DIR, f"N_{N}__PAR__{task}__{safe_model}__{dialect}__{condition}.json")

def log_path(task, model, dialect):
    safe_model = model.replace("/", "_")
    return os.path.join(LOG_DIR, f"N_{N}__PAR__{task}__{safe_model}__{dialect}.json")
def write_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

################################## Running an experiment ##################################

def run_one(task, model, dialect):
    examples = load_task(task, dialect, N, SEED)
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
                desc=f"{task} | {model} | {dialect} | {condition} | PAR",
                leave=False,
            )
        ):
            dialect_text = get_dialect_text(task, ex)
            dialect_prompt = build_prompt_from_paraphrase(task, ex, dialect_text)
            paraphrase_prompt = build_paraphrase_prompt(dialect_text, dialect)

            para_start = time.time()
            paraphrase_raw = call_model(paraphrase_prompt, model)
            para_latency = time.time() - para_start

            prompt = build_prompt_from_paraphrase(task, ex, paraphrase_raw.strip())
            time.sleep(0.05)

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

            log_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "task": task,
                "model": model,
                "dialect": dialect,
                "condition": condition,
                "example_index": i,
                "latency_seconds": latency,
                "dialect_prompt": dialect_prompt,
                "prompt": prompt,
                "response": raw,
            }
            if paraphrase_prompt is not None:
                log_entry["paraphrase_prompt"] = paraphrase_prompt
                log_entry["paraphrase_response"] = paraphrase_raw
                log_entry["paraphrase_latency_seconds"] = para_latency
            log_rows.append(log_entry)

            correct += 1 if is_correct else 0
            total += 1

            time.sleep(0.05)

        write_json(out_path, results_rows)

        acc = correct / total if total else 0.0
        results[condition] = {"accuracy": acc, "correct": correct, "total": total}

    write_json(log_file, log_rows)

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--dialect", required=True, choices=DIALECTS)
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()

    ensure_dirs()
    summary = []

    res = run_one(args.task, args.model, args.dialect)
    summary.append({
        "task": args.task,
        "dialect": args.dialect,
        "model": args.model,
        "dialect_accuracy": res[args.dialect]["accuracy"],
    })
    print(args.task, args.dialect, args.model, "DIALECT", res[args.dialect]["accuracy"])

    with open(os.path.join(OUT_DIR, f"N_{N}__summary_paraphrase.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
