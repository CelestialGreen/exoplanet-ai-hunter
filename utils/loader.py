import json

def load_env(filepath="ENV.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data