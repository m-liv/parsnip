# Tasks:
# Language Understanding: BoolQ
# Algorithmic Understanding: MBPP
# Math: GSM8K
# Logic: FOLIO
TASKS = ["boolq", "mbpp", "gsm8k", "folio"]

BOOLQ_SIZES = {
    "AAVE": 917,
    "IndE": 400
}

# Dialects:
# AAVE: African American Vernacular English
# IndE: Indian English
DIALECTS = ["AAVE", "IndE", "JamE", "CollSgE", "ChcE"]  
   
# Models:
# Gemini 2.5 Pro: native reasoning model
# GPT-4o: mid-tier model
# GPT-4o mini: small model
MODELS = ["gemini-2.5-pro", "gpt-4o", "gpt-4o-mini"]     
  
# Number of examples, for each task-dialect-model combination
#N = 200      
N = 30 # for testing           
      
# Random seed
SEED = 2

# Output and log directories for baseline results
OUT_DIR = "results/baseline"
LOG_DIR = "logs/baseline"