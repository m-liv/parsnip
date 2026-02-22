from openai import OpenAI
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env (API keys)

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# resp = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "Say hello"}],
#     temperature=0
# )
# print(resp.choices[0].message.content)

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# resp = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Say hello"}],
#     temperature=0
# )
# print(resp.choices[0].message.content)

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])
# model = genai.GenerativeModel("gemini-2.5-pro")
# response = model.generate_content("Say hello")
# print(response.text)