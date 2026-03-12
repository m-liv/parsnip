import re
import io
import contextlib
################################## Parsing model output ##################################
# Parse WSC model output into integer
def parse_wsc(text):
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    t = nums[-1] if nums else None
    # print("Parsed WSC output:", t)
    if t is not None:
        if t == "1":
            return 1
        if t == "0":
            return 0
    return None


# Parse MultiRC model output into integer choice (1 or 0)
def parse_multirc(text):
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    t = nums[-1] if nums else None
    # print("Parsed MultiRC output:", t)
    if t is not None:
        if t == "1":
            return 1
        if t == "0":
            return 0
    return None

# Parse Logic Bench MCQ model output into integer choice (1, 2, 3, or 4)
def parse_logic_bench_mcq(text):
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    t = nums[-1] if nums else None
    #print("Parsed Logic Bench MCQ output:", t)
    if t is not None:
        if t in ["1", "2", "3", "4"]:
            return int(t)
    return None

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
    # print("Original text:", text)
    num = nums[-1] if nums else None
    if num is not None:
        try:
            # Remove anything after a decimal point
            if "." in num:
                num = num.split(".")[0]
                # print("Extracted numbers:", num)
            return int(num)
        except ValueError:
            return None

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
    # Redirect stdout and stderr to block printing from the executed code
    buf_out = io.StringIO()
    buf_err = io.StringIO()

    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exec(code, local_env, local_env)
    except Exception as e:
        # print("Code execution error:", repr(e))
        # traceback.print_exc()
        return False

    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exec(test_cases, local_env, local_env)
    except Exception as e:
        # print("Test cases execution error:", repr(e))
        # traceback.print_exc()
        return False

    return True
    

# Parse model output for a given task into appropriate format for scoring
def parse_task(task, text):
    if task == "wsc":
        return parse_wsc(text)
    if task == "multirc":
        return parse_multirc(text)
    if task == "logic_bench_mcq":
        return parse_logic_bench_mcq(text)
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
    # For WSC, check if parsed prediction matches integer label (1 or 0)
    if task == "wsc":
        # print("Parsed prediction:", pred)
        # print("Integer label:", label)
        #print("Comparison:", pred, "==", label)
        return pred == label
    
    # For MultiRC, check if parsed prediction matches integer label (1 or 0)
    if task == "multirc":
        # print("Parsed prediction:", pred)
        # print("Integer label:", label)
        # print("Comparison:", pred, "==", label)
        return pred == int(label)
    
    # For Logic Bench MCQ, check if parsed prediction matches integer label (1, 2, 3, or 4)
    if task == "logic_bench_mcq":
        # print("Parsed prediction:", pred)
        # print("Integer label:", label)
        # print("Comparison:", pred, "==", label)
        return pred == int(label)
    
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
        # print("Parsed prediction:", pred)
        # print("Numeric label:", label)
        # Parse label into numeric value: remove commas and convert to int
        num_label = int(label.replace(",", ""))
        # print("Comparison:", pred, "==", num_label)
        return pred == num_label
    
    # For FOLIO, check if parsed prediction matches string label (True, False, or Uncertain)
    if task == "folio":
        return pred == label
   
    return False
