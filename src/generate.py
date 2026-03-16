import yaml
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load config
def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except:
            logger.info(f"Error reading yaml file")
            return None
config = load_config('configs/tiny_llm.yaml')

# Set Logger
log_dir = config["paths"]["log_dir"]
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
# Load Tokenizer, Model
checkpoint_dir = config["paths"]["checkpoint_dir"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

model.to(device)
model.eval()

# Tokenize
prompt = config["generation"]["prompt"]
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=config["generation"]["max_new_tokens"],
    )
    
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)