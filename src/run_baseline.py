import os
import json
import time
import re
from datasets import load_dataset
from config import TASKS, DIALECTS, MODELS, N, SEED, OUT_DIR

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
        return ex["sae_prompt"] if condition == "SAE" else ex["dialect_prompt"]
    
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