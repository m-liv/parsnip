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

