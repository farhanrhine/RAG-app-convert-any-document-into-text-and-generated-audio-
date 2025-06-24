# # utils.py
# import json
# from langchain_core.prompts import PromptTemplate

# def load_prompt(filename):
#     with open(filename, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     return PromptTemplate.from_template(data['template'])



# utils.py

import json
from langchain_core.prompts import PromptTemplate

def load_prompt(filename):
    """
    Loads a prompt template from a JSON file and returns a PromptTemplate object.

    Why this is useful:
    - Keeps prompt loading logic separate from the main UI code (clean code principle).
    - Allows easy reuse of prompt-loading in other parts of the app.
    - Makes it easy to update or change prompt formats later without touching the UI code.
    - Supports future expansion (e.g., multiple prompt types, dynamic variables, etc.).
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert the loaded template string into a LangChain PromptTemplate
    return PromptTemplate.from_template(data['template'])
