# Tasks:
# Language Understanding: WSC, MultiRC
# Logic: Logic Bench MCQ, FOLIO
TASKS = ["wsc", "multirc", "logic_bench_mcq", "folio"]

# Dialects:
# AAVE: African American Vernacular English
# IndE: Indian English
DIALECTS = ["AAVE", "IndE"]  
   
# Models:
# Gemini 2.5 Pro: native reasoning model
# GPT-4o: mid-tier model
# GPT-4o mini: small model
MODELS = ["gemini-2.5-pro", "gpt-4o", "gpt-4o-mini"]     
  
# Specify sample size for current iteration
N = 200      
 
# Total available task sizes:
# WSC: AAVE = 580, IndE = 511
# MultiRC: AAVE = 986, IndE = 376
# Logic Bench MCQ: AAVE = 480, IndE = 239
# FOLIO: AAVE = 925, IndE = 298        
      
# Random seed
SEED = 20
