import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
import logging
import os

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

# Model name
model_name = "distilgpt2"

# Load tokenizer and model
logger.info("Load model")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
logger.info("Load dataset")
dataset = load_from_disk("datasets/raw/wikitext-2-raw-v1")

# Tokenize dataset
def tokenize_fn(batch):
    return tokenizer(batch["text"])

logger.info("Tokenize dataset")
tokenized_dataset = dataset.map(tokenize_fn,
                                batched=True,
                                batch_size=1000,
                                remove_columns=["text"],
                                )

# Save tokenized data
logger.info("Save tokenized dataset")
tokenized_dataset.save_to_disk("datasets/tokenized/wikitext-2-raw-v1_tokenized")

# Load tokenized data
logger.info("Load tokenized dataset")
dataset = load_from_disk("datasets/tokenized/wikitext-2-raw-v1_tokenized")

# Block-wise chunking of tokenized sequences for training
def group_texts(dataset, block_size):
    dataset = {
        k: sum(dataset[k], [])
        for k in dataset.keys()
    }
    
    total_length = len(dataset["input_ids"])
    total_length = (total_length // block_size) * block_size
    
    result = {
        k: [
            t[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        for k, t in dataset.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

block_size = config["train"]["block_size"]

logger.info("Chunking dataset to block-wise")
block_dataset = dataset.map(
    lambda x: group_texts(x, block_size),
    batched=True,
)

logger.info("Save block dataset to disk")
block_dataset.save_to_disk(f"datasets/block_wise/wikitext-2-raw-v1_block_size_{block_size}")

# Training arguments, Trainer
training_args = TrainingArguments(
    output_dir=config["paths"]["output_root"],
    overwrite_output_dir=True,

    per_device_train_batch_size=config["train"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["train"]["per_device_eval_batch_size"],

    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=config["train"]["eval_steps"],
    save_steps=config["train"]["save_steps"],
    logging_steps=config["train"]["logging_steps"],

    max_steps=config["train"]["max_steps"],
    learning_rate=config["train"]["learning_rate"],
    weight_decay=config["train"]["weight_decay"],

    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=block_dataset["train"],
    eval_dataset=block_dataset["validation"],
    processing_class=tokenizer,
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = "right"

logger.info("Train model")
trainer.train()

logger.info("Save model to disk")
checkpoint_dir = config["paths"]["checkpoint_dir"]
trainer.save_model(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)