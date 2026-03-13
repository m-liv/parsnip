import json
import os
from statsmodels.stats.contingency_tables import mcnemar

def load_results(path):
    """Load JSON results and return dict mapping example index -> correct (bool)."""
    with open(path) as f:
        rows = json.load(f)
    return {r["i"]: bool(r["correct"]) for r in rows}

def run_mcnemar(sae_path, dialect_path):
    """Run McNemar's test on paired SAE vs dialect results."""
    sae = load_results(sae_path)
    dia = load_results(dialect_path)
    
    # Only include examples present in both
    indices = sorted(set(sae.keys()) & set(dia.keys()))
    
    # Build 2x2 table
    a = sum(sae[i] and dia[i] for i in indices)       # both correct
    b = sum(sae[i] and not dia[i] for i in indices)    # SAE correct, dialect wrong
    c = sum(not sae[i] and dia[i] for i in indices)    # SAE wrong, dialect correct
    d = sum(not sae[i] and not dia[i] for i in indices)# both wrong
    
    n = len(indices)
    table = [[a, b], [c, d]]
    
    # Use exact test when discordant pairs < 25
    exact = (b + c) < 25
    result = mcnemar(table, exact=exact)
    
    return {
        "n": n,
        "both_correct": a,
        "sae_only": b,
        "dialect_only": c,
        "both_wrong": d,
        "p_value": result.pvalue,
        "significant": result.pvalue < 0.05,
    }

# === Run for all your completed experiments ===

base_dir = "results/baseline"  # change to "results/dialect_aware" if you want to analyze those results instead

experiments = [
    # (task, model, dialect)
    ("wsc", "gpt-4o-mini", "AAVE"),
    ("wsc", "gpt-4o-mini", "IndE"),
    ("wsc", "gpt-4o", "AAVE"),
    ("wsc", "gpt-4o", "IndE"),
    ("multirc", "gpt-4o-mini", "AAVE"),
    ("multirc", "gpt-4o-mini", "IndE"),
    ("multirc", "gpt-4o", "AAVE"),
    ("multirc", "gpt-4o", "IndE"),
    ("logic_bench_mcq", "gpt-4o-mini", "AAVE"),
    ("logic_bench_mcq", "gpt-4o-mini", "IndE"),
    ("logic_bench_mcq", "gpt-4o", "AAVE"),
    ("logic_bench_mcq", "gpt-4o", "IndE"),
    ("folio", "gpt-4o-mini", "AAVE"),
    ("folio", "gpt-4o-mini", "IndE"),
    ("folio", "gpt-4o", "AAVE"),
    ("folio", "gpt-4o", "IndE"),
]

print(f"{'Task':<16} {'Model':<14} {'Dialect':<6}  N    b    c    p-value   sig?")
print("-" * 75)

for task, model, dialect in experiments:
    safe_model = model.replace("/", "_")
    sae_file = os.path.join(base_dir, f"{task}__{safe_model}__{dialect}__SAE.json")
    dia_file = os.path.join(base_dir, f"{task}__{safe_model}__{dialect}__{dialect}.json")
    
    if not os.path.exists(sae_file) or not os.path.exists(dia_file):
        print(f"{task:<16} {model:<14} {dialect:<6}  -- files not found, skipping")
        continue
    
    r = run_mcnemar(sae_file, dia_file)
    sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
    print(f"{task:<16} {model:<14} {dialect:<6}  {r['n']:<4} {r['sae_only']:<4} {r['dialect_only']:<4} {r['p_value']:<9.4f} {sig}")
