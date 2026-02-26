################################## Building prompts ##################################

# Build prompt for a given task, example, and condition (SAE or dialect)
def build_prompt(task, ex, condition):
    # For BoolQ, ask for T/F answer to question based on passage
    if task == "boolq":
        passage = ex["sae_passage"] if condition == "SAE" else ex["dialect_passage"]
        question = ex["sae_question"]
        return (
            f"Passage: \"{passage}\"\n"
                f"Question: \"{question}\"\n"
                "Is the answer TRUE or FALSE?\nAnswer:"
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