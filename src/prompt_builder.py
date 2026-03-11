################################## Building prompts ##################################

# Build prompt for a given task, example, and condition (SAE or dialect)
def build_prompt(task, ex, condition):
    # For WSC, ask whether pronoun refers to target noun
    if task == "wsc":
        paragraph = ex["sae_paragraph"] if condition == "SAE" else ex["dialect_paragraph"]
        span1 = ex["span_1"]
        span2 = ex["span_2"]
        return (
            f"Check if Span 2 refers to Span 1 in the paragraph.\n"
                f"Paragraph: {paragraph}\nSpan 1: {span1}\nSpan 2: {span2}\n"
                "Answer (1 if same, 0 if not):"
        )
        
    # For Logic Bench MCQ, ask for choice between 4 options based on context
    if task == "logic_bench_mcq":
        context = ex["sae_input"] if condition == "SAE" else ex["dialect_input"]
        choices = [ex[f"choice_{i}"] for i in range(1, 5)]
        return (
            f"Select the correct choice from 1, 2, 3, or 4.\n"
                f"Context: {context}\nChoice 1: {choices[0]}\nChoice 2: {choices[1]}\n"
                f"Choice 3: {choices[2]}\nChoice 4: {choices[3]}\n"
                "Answer (1, 2, 3, or 4):"
        )
    
    # For MultiRC, ask for 1/0 answer to question based on passage
    if task == "multirc":
        paragraph = ex["sae_input"] if condition == "SAE" else ex["dialect_input"]
        question = ex["question"]
        answer_choice = ex["answer_choice"]
        return (
            f"Given a paragraph, a question, and an answer choice, is the choice correct (1) or incorrect (0)?\n"
                f"Paragraph: {paragraph}\nQuestion: {question}\nAnswer Choice: {answer_choice}\n"
                "Answer:"
        )
    
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