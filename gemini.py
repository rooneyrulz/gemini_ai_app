import os
import json

import google.generativeai as genai

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

config_file_path = f"{working_dir}/config.json"
config_data = json.load(open(config_file_path))

GOOGLE_API_KEY = config_data["google_api_key"]

genai.configure(api_key=GOOGLE_API_KEY)

def load_model():
    genimi_pro_model = genai.GenerativeModel("gemini-pro")
    return genimi_pro_model

def load_vision_model(prompt, image):
    vision_model = genai.GenerativeModel("gemini-1.5-flash-001")
    response = vision_model.generate_content([prompt, image])
    return response.text

def load_embedding_model(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(
        model=embedding_model,
        content=input_text,
        task_type="retrieval_document")
    return embedding["embedding"]

def load_qa_model(user_prompt):
    qa_model = genai.GenerativeModel("gemini-pro")
    response = qa_model.generate_content(user_prompt)
    return response.text
