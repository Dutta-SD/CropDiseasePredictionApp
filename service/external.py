import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
SYS_INSTR = "You are a plant disease expert. You will be given an image of a plant and you will give the remedy for the disease in 100 words. Respond in Markdown."
PROMPT = "Suggest remedy for the disease {}"

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=SYS_INSTR)
generation_config = genai.GenerationConfig(
    max_output_tokens=1000,
    temperature=0.1,
)


def get_gemini_response(input_text):
    prompt = PROMPT.format(input_text)
    response = model.generate_content(prompt, generation_config=generation_config)
    return response.text
