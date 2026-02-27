from datasets import load_dataset

################################## Loading examples from datasets ##################################
# Load WSC examples for a given dialect, number of examples, and random seed
def load_wsc(dialect, n, seed):
    ds = load_dataset("abhaygupta1266/wsc")
    split = ds[dialect].shuffle(seed=seed).select(range(n))
    
    examples = []
    
    for e in split:
        examples.append({
            "sae_paragraph": e["Original Paragraph"],
            "span_1": e["Span 1"],
            "span_2": e["Span 2"],
            "dialect_paragraph": e["Dialect (Original Paragraph)"],
            "label": e["Actual Label"],
            "bleu": e["BLEU Score Original Paragraph"]
        })
        
    return examples

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
            "label": e["Answer"],
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
    if task == "wsc":
        return load_wsc(dialect, n, seed)
    if task == "boolq":
        return load_boolq(dialect, n, seed)
    if task == "gsm8k":
        return load_gsm8k(dialect, n, seed)
    if task == "mbpp":
        return load_mbpp(dialect, n, seed)
    if task == "folio":
        return load_folio(dialect, n, seed)
    return []
