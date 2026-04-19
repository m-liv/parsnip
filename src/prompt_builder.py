################################## Building prompts ##################################

# Build baseline prompt (no intervation strategy) for a given task, example, and condition (SAE or dialect)
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

# Build prompt with dialect-aware instructions for a given task, example, and condition (SAE or dialect)
def build_dialect_aware_prompt(task, ex, condition, dialect):
    if condition == "SAE":
        dialect_name = "English"
    elif dialect == "AAVE":
        dialect_name = "the African American Vernacular English (AAVE) dialect"
    elif dialect == "IndE":
        dialect_name = "the Indian English dialect"
    # For WSC, ask whether pronoun refers to target noun
    if task == "wsc":
        paragraph = ex["sae_paragraph"] if condition == "SAE" else ex["dialect_paragraph"]
        span1 = ex["span_1"]
        span2 = ex["span_2"]
        return (
            f"You are given a paragraph written in {dialect_name}.\nCheck if Span 2 refers to Span 1 in the paragraph.\n"
                f"Paragraph: {paragraph}\nSpan 1: {span1}\nSpan 2: {span2}\n"
                "Answer (1 if same, 0 if not):"
        )
        
    # For Logic Bench MCQ, ask for choice between 4 options based on context
    if task == "logic_bench_mcq":
        context = ex["sae_input"] if condition == "SAE" else ex["dialect_input"]
        choices = [ex[f"choice_{i}"] for i in range(1, 5)]
        return (
            f"You are given a question written in {dialect_name}. Select the correct choice from 1, 2, 3, or 4.\n"
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
            f"You are given a paragraph written in {dialect_name}.\nGiven a paragraph, a question, and an answer choice, is the choice correct (1) or incorrect (0)?\n"
                f"Paragraph: {paragraph}\nQuestion: {question}\nAnswer Choice: {answer_choice}\n"
                "Answer:"
        )
    
    
    # For FOLIO, ask for T/F answer to conclusion based on premises
    if task == "folio":
        premises = ex["sae_input"] if condition == "SAE" else ex["dialect_input"]
        conclusion = ex["conclusion"]
        return (
            f"You are given premises written in {dialect_name}.\nDetermine whether the following conclusion is true, false, or uncertain based on the premises. Answer with only True, False, or Uncertain.\n\n"
            f"Premises: {premises}\n"
            f"Conclusion: {conclusion}\n"
            "Answer:"
        )

    return ""

################################## Few-shot helpers ##################################

def _format_few_shot_examples(task, shots):
    """Render the list of shot dicts into a string of example turns."""
    lines = []
    for shot in shots:
        if task == "wsc":
            lines.append(
                f"Paragraph: {shot['paragraph']}\nSpan 1: {shot['span_1']}\nSpan 2: {shot['span_2']}\n"
                f"Answer (1 if same, 0 if not): {shot['answer']}"
            )
        elif task == "logic_bench_mcq":
            lines.append(
                f"Context: {shot['context']}\nChoice 1: {shot['choice_1']}\nChoice 2: {shot['choice_2']}\n"
                f"Choice 3: {shot['choice_3']}\nChoice 4: {shot['choice_4']}\n"
                f"Answer (1, 2, 3, or 4): {shot['answer']}"
            )
        elif task == "multirc":
            lines.append(
                f"Paragraph: {shot['paragraph']}\nQuestion: {shot['question']}\nAnswer Choice: {shot['answer_choice']}\n"
                f"Answer: {shot['answer']}"
            )
        elif task == "folio":
            lines.append(
                f"Premises: {shot['premises']}\nConclusion: {shot['conclusion']}\n"
                f"Answer: {shot['answer']}"
            )
    return "\n\n".join(lines)

def build_few_shot_prompt(task, ex, dialect):
    """Build a task prompt prefixed with 3 dialectal few-shot examples."""
    from few_shot_examples import FEW_SHOT_EXAMPLES
    shots = FEW_SHOT_EXAMPLES.get(task, {}).get(dialect, [])
    examples_block = _format_few_shot_examples(task, shots)

    if task == "wsc":
        paragraph = ex["dialect_paragraph"]
        span1 = ex["span_1"]
        span2 = ex["span_2"]
        return (
            "Check if Span 2 refers to Span 1 in the paragraph. Here are some examples:\n\n"
            f"{examples_block}\n\n"
            "Now answer the following:\n"
            f"Paragraph: {paragraph}\nSpan 1: {span1}\nSpan 2: {span2}\n"
            "Answer (1 if same, 0 if not):"
        )
    if task == "logic_bench_mcq":
        context = ex["dialect_input"]
        choices = [ex[f"choice_{i}"] for i in range(1, 5)]
        return (
            "Select the correct choice from 1, 2, 3, or 4. Here are some examples:\n\n"
            f"{examples_block}\n\n"
            "Now answer the following:\n"
            f"Context: {context}\nChoice 1: {choices[0]}\nChoice 2: {choices[1]}\n"
            f"Choice 3: {choices[2]}\nChoice 4: {choices[3]}\n"
            "Answer (1, 2, 3, or 4):"
        )
    if task == "multirc":
        paragraph = ex["dialect_input"]
        question = ex["question"]
        answer_choice = ex["answer_choice"]
        return (
            "Given a paragraph, a question, and an answer choice, is the choice correct (1) or incorrect (0)? "
            "Here are some examples:\n\n"
            f"{examples_block}\n\n"
            "Now answer the following:\n"
            f"Paragraph: {paragraph}\nQuestion: {question}\nAnswer Choice: {answer_choice}\n"
            "Answer:"
        )

    if task == "folio":
        premises = ex["dialect_input"]
        conclusion = ex["conclusion"]
        return (
            "Determine whether the following conclusion is true, false, or uncertain based on the premises. "
            "Answer with only True, False, or Uncertain. Here are some examples:\n\n"
            f"{examples_block}\n\n"
            "Now answer the following:\n"
            f"Premises: {premises}\nConclusion: {conclusion}\n"
            "Answer:"
        )
    return ""

################################## Paraphrase helpers ##################################

def get_dialect_text(task, ex):
    """Return the dialectal text for a given task example."""
    fields = {
        "wsc": "dialect_paragraph",
        "logic_bench_mcq": "dialect_input",
        "multirc": "dialect_input",
        "folio": "dialect_input",
    }
    key = fields.get(task)
    return ex.get(key, "") if key else ""

def build_paraphrase_prompt(dialect_text, dialect):
    if dialect == "AAVE":
        dialect_name = "African American Vernacular English (AAVE)"
    elif dialect == "IndE":
        dialect_name = "Indian English"
    else:
        dialect_name = dialect
    return (
        f"Paraphrase the following text from {dialect_name} into Standard American English. "
        "Preserve the original meaning exactly. Output only the paraphrased text, nothing else.\n\n"
        f"Text: {dialect_text}\n"
        "Paraphrase:"
    )

def build_prompt_from_paraphrase(task, ex, paraphrased_text):
    """Build the task prompt using paraphrased SAE text in place of the original dialect text."""
    if task == "wsc":
        span1 = ex["span_1"]
        span2 = ex["span_2"]
        return (
            "Check if Span 2 refers to Span 1 in the paragraph.\n"
            f"Paragraph: {paraphrased_text}\nSpan 1: {span1}\nSpan 2: {span2}\n"
            "Answer (1 if same, 0 if not):"
        )
    if task == "logic_bench_mcq":
        choices = [ex[f"choice_{i}"] for i in range(1, 5)]
        return (
            "Select the correct choice from 1, 2, 3, or 4.\n"
            f"Context: {paraphrased_text}\nChoice 1: {choices[0]}\nChoice 2: {choices[1]}\n"
            f"Choice 3: {choices[2]}\nChoice 4: {choices[3]}\n"
            "Answer (1, 2, 3, or 4):"
        )
    if task == "multirc":
        question = ex["question"]
        answer_choice = ex["answer_choice"]
        return (
            "Given a paragraph, a question, and an answer choice, is the choice correct (1) or incorrect (0)?\n"
            f"Paragraph: {paraphrased_text}\nQuestion: {question}\nAnswer Choice: {answer_choice}\n"
            "Answer:"
        )
    if task == "folio":
        conclusion = ex["conclusion"]
        return (
            "Determine whether the following conclusion is true, false, or uncertain based on the premises. "
            "Answer with only True, False, or Uncertain.\n\n"
            f"Premises: {paraphrased_text}\n"
            f"Conclusion: {conclusion}\n"
            "Answer:"
        )
    return ""