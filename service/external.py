import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

SYS_INSTR = "You are a plant disease expert. You will be given queries regarding plant diseases. Always respond in Markdown"
TXT_PROMPT = "Suggest remedy for the disease in bullet points"
IMG_TXT_PROMPT = "Based on the given image, suggest the possible disease the plant is suffering from, along with the remedy in 150 words"


def llm_strategy(llm_name, disease_name, image_file=None):
    if llm_name.lower() == "gemini":
        return get_response_from_gemini(disease_name, image_file)
    else:
        raise ValueError(f"LLM {llm_name} not supported")


def get_response_from_gemini(disease_name, image_file=None) -> str:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=SYS_INSTR)

    generation_config = genai.GenerationConfig(max_output_tokens=300)

    prompt = [TXT_PROMPT, disease_name]
    if image_file:
        prompt = [IMG_TXT_PROMPT, image_file]
    response = model.generate_content(prompt, generation_config=generation_config)
    return response.text
