import time
import os
from openai import OpenAI
from google import genai

################################## Calling models ##################################
def call_openai(prompt, model, retries=3, backoff_factor=2):
    client = OpenAI()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait_time = backoff_factor ** attempt
            print(f"API error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    raise Exception("Maximum retries exceeded.")

def call_gemini(prompt, model):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text

def call_model(prompt, model):
    if model.startswith("gpt-"):
        return call_openai(prompt, model)
    return call_gemini(prompt, model)