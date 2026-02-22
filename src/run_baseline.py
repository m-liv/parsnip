import os
import json
import time
import re
import datetime
from tqdm import tqdm
from itertools import product
from datasets import load_dataset
from openai import OpenAI
from google import genai
from config import TASKS, DIALECTS, MODELS, N, SEED, OUT_DIR, LOG_DIR
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env (API keys)
from huggingface_hub import login

if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

################################## Loading examples from datasets ##################################

# Load BoolQ examples for a given dialect, number of examples, and random seed
def load_boolq(dialect, n, seed):
    ds = load_dataset("abhaygupta1266/boolq")
    split = ds[dialect].shuffle(seed=seed).select(range(n))
    
    examples = []
    
    for e in split:
        examples.append({
            "sae_passage": e["SAE Passage"],
            "sae_question": e["SAE Question"],
            "dialect_passage": e["Dialect (SAE Passage)"],
            "label": e["Actual Label"],
            "bleu": e["BLEU Score SAE Passage"]
        })
        
    return examples

# Load MBPP examples for a given dialect, number of examples, and random seed
def load_mbpp(dialect, n, seed):
    ds = load_dataset("abhaygupta1266/mbpp")
    split = ds[dialect].shuffle(seed=seed).select(range(n))
    
    examples = []
    
    for e in split:
        examples.append({
            "sae_prompt": e["Original"],
            "dialect_prompt": e["Dialect (Original)"],
            "reference_code": e["Code"],
            "test_cases": e["Test_Cases"],
            "bleu": e["BLEU Score Original"],
        })
        
    return examples

# Load GSM8K examples for a given dialect, number of examples, and random seed
def load_gsm8k(dialect, n, seed):
    ds = load_dataset("abhaygupta1266/gsm8k")
    split = ds[dialect].shuffle(seed=seed).select(range(n))
    
    examples = []
    
    for e in split:
        examples.append({
            "sae_question": e["Original"],
            "dialect_question": e["Dialect (Original)"],
            "sae_answer": e["Answer"],
            "bleu": e["BLEU Score Original"],
        })
        
    return examples

# Load FOLIO examples for a given dialect, number of examples, and random seed
def load_folio(dialect, n, seed):
    ds = load_dataset("abhaygupta1266/folio")
    split = ds[dialect].shuffle(seed=seed).select(range(n))
    
    examples = []
    
    for e in split:
        examples.append({
            "sae_input": e["Premises"],
            "dialect_input": e["Dialect (Premises)"],
            "conclusion": e["Conclusion"],
            "label": e["Label"],
            "bleu": e["BLEU Score Premises"],
        })
        
    return examples

# Load examples for a given task, dialect, number of examples, and random seed
def load_task(task, dialect, n, seed):
    if task == "boolq":
        return load_boolq(dialect, n, seed)
    if task == "gsm8k":
        return load_gsm8k(dialect, n, seed)
    if task == "mbpp":
        return load_mbpp(dialect, n, seed)
    if task == "folio":
        return load_folio(dialect, n, seed)
    return []


################################## Building prompts ##################################

# Build prompt for a given task, example, and condition (SAE or dialect)
def build_prompt(task, ex, condition):
    # For BoolQ, ask for T/F answer to question based on passage
    if task == "boolq":
        passage = ex["sae_passage"] if condition == "SAE" else ex["dialect_passage"]
        question = ex["sae_question"]
        return (
            "Answer the following question with only True or False.\n\n"
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            "Answer:"
        )
    
    # For GSM8K, ask for numeric answer to math problem
    if task == "gsm8k":
        q = ex["sae_question"] if condition == "SAE" else ex["dialect_question"]
        return (
            "Solve the problem. Provide only the final numeric answer.\n\n"
            f"{q}\n"
            "Answer:"
        )
        
    # For MBPP, provide the code prompt
    if task == "mbpp":
        q = ex["sae_prompt"] if condition == "SAE" else ex["dialect_prompt"]
        test_cases = ex["test_cases"]
        return (
            "Given a coding problem, produce a Python function.\n\n"
            f"Start it with 'Answer:' on its own line.\n"
            f"Problem: {q}\nTest Cases: {test_cases}\nAnswer:"
        )
    
    # For FOLIO, ask for T/F answer to conclusion based on premises
    if task == "folio":
        premises = ex["sae_input"] if condition == "SAE" else ex["dialect_input"]
        conclusion = ex["conclusion"]
        return (
            "Determine whether the following conclusion is true, false, or uncertain based on the premises. Answer with only True, False, or Uncertain.\n\n"
            f"Premises: {premises}\n"
            f"Conclusion: {conclusion}\n"
            "Answer:"
        )
        
    return ""

################################## Parsing model output ##################################

# Parse BoolQ model output into boolean value (True, False, or None if unclear)
def parse_boolq(text):
    t = text.strip().lower()
    if "true" in t and "false" not in t:
        return True
    if "false" in t and "true" not in t:
        return False
    return None

# Parse GSM8K model output into numeric value (last number in output, or None if no numbers)
def parse_gsm8k(text):
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None

# Parse FOLIO model output into string value (True, False, Uncertain, or None if unclear)
def parse_folio(text):
    t = text.strip().lower()
    if "true" in t and "false" not in t and "uncertain" not in t:
        return "True"
    if "false" in t and "true" not in t and "uncertain" not in t:
        return "False"
    if "uncertain" in t and "true" not in t and "false" not in t:
        return "Uncertain"
    return None

# For MBPP, extract the code block from the model output (or return full output if no code block)
def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1)
    else:
        code = text

    code = re.sub(r"^\s*Answer\s*:\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"```(?:python)?", "", code)
    code = re.sub(r"```", "", code)

    return code.strip()

# Assess MBPP model output by executing the extracted code and checking if it runs without error and passes all test cases
import traceback
def assess_mbpp(code, test_cases):
    local_env = {}
    try:
        exec(code, local_env, local_env)
    except Exception as e:
        # print("Code execution error:", repr(e))
        # traceback.print_exc()
        return False

    try:
        exec(test_cases, local_env, local_env)
    except Exception as e:
        # print("Test cases execution error:", repr(e))
        # traceback.print_exc()
        return False

    return True
    

# Parse model output for a given task into appropriate format for scoring
def parse_task(task, text):
    if task == "boolq":
        return parse_boolq(text)
    if task == "gsm8k":
        return parse_gsm8k(text)
    if task == "folio":
        return parse_folio(text)
    if task == "mbpp":
        return text
    return text


################################## Scoring model output ##################################

# Compute score for a given task by comparing model prediction to label, using appropriate parsing and assessment for each task
def score(task, pred, label, example):
    # For BoolQ, check if parsed prediction matches boolean label (True or False)
    if task == "boolq":
        return pred == label
    
    # For MBPP, check if code runs without error and passes test cases
    if task == "mbpp":
        code = extract_code(pred)
        # print("=" * 100)
        # print("PROMPT:", example["sae_prompt"])
        # print()
        # print("MODEL's CODE (cleaned):", code)
        # print()
        # print("REFERENCE CODE:", example["reference_code"])
        # print()
        # print("TEST CASES:", example["test_cases"])
        # print()
        results = assess_mbpp(code, example["test_cases"])
        # print("RESULTS - Passed:", results)
        # print("=" * 100)
        # print()
        return results
    
    # For GSM8K, check if parsed prediction matches numeric label (extracted from text)
    if task == "gsm8k":
        return pred == label
    
    # For FOLIO, check if parsed prediction matches string label (True, False, or Uncertain)
    if task == "folio":
        return pred == label
   
    return False


################################## Calling models ##################################
def call_openai(prompt, model, retries=3, backoff_factor=2):
    client = OpenAI()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait_time = backoff_factor ** attempt
            print(f"API error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    raise Exception("Maximum retries exceeded.")

def call_gemini(prompt, model):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text

def call_model(prompt, model):
    if model.startswith("gpt-"):
        return call_openai(prompt, model)
    return call_gemini(prompt, model)


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
    d_tasks = ["mbpp"]
    d_dialects = ["AAVE", "IndE"]
    d_models = ["gpt-4o-mini"]
    
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