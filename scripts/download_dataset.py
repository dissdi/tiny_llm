from datasets import load_dataset, DatasetDict
import os

# Download and load the wikitext-2-raw-v1 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Access the splits
train_data = dataset['train']
validation_data = dataset['validation']
test_data = dataset['test']

# Save dataset
dataset_dict = DatasetDict({
    "train": train_data,
    "validation": validation_data,
    "test": test_data,
})

dataset_dict.save_to_disk("datasets/raw/wikitext-2-raw-v1")