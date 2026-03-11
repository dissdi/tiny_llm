import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk


# Load config
def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except:
            print(f"Error reading yaml file")
            return None

config_data = load_config('configs/tiny_llm.yaml')

# Model name
model_name = "distilgpt2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_from_disk("datasets/wikitext-2-raw-v1")

