import json
import os

from datasets import load_dataset
import torch
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import logging

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

MAX_LEN = 384
BATCH_SIZE = 8
DATA_PATH = "prepared_pairs.pt"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def split_prompt_answer(text: str):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    assistant_indices = [i for i, line in enumerate(lines) if line.startswith("Assistant:")]
    if not assistant_indices:
        return text, ""
    last_idx = assistant_indices[-1]
    prompt_lines = lines[:last_idx] + ["Assistant:"]
    prompt = "\n".join(prompt_lines)
    answer_lines = [lines[last_idx][len("Assistant:"):].strip()] + lines[last_idx+1:]
    answer = "\n".join(answer_lines).strip()
    return prompt, answer

def format_dpo_example(example):
    chosen_prompt, chosen_answer = split_prompt_answer(example["chosen"])
    rejected_prompt, rejected_answer = split_prompt_answer(example["rejected"])
    return {
        "prompt_formatted": chosen_prompt,
        "chosen_formatted": chosen_answer,
        "rejected_formatted": rejected_answer,
    }

def tokenize_pair(prompt, answer, tokenizer, max_len=MAX_LEN):
    enc = tokenizer(
        prompt, answer,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {
        "input_ids": enc.input_ids.squeeze(0),
        "attention_mask": enc.attention_mask.squeeze(0)
    }

def prepare_and_save_dataset(hf_dataset, tokenizer):
    # Форматируем
    formatted = hf_dataset.map(format_dpo_example)

    # Делим на train/val
    splitted = formatted.train_test_split(test_size=0.1)
    train_raw = splitted["train"]
    val_raw = splitted["test"]

    # Сохраняем валид. данные для eval
    val_raw_simple = val_raw.remove_columns(
        [col for col in val_raw.column_names if col not in ("prompt_formatted", "chosen_formatted")]
    )

    val_raw_simple = val_raw_simple.to_dict()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(save_dir, exist_ok=True)

    val_raw_path = os.path.join(save_dir, "val_raw.json")

    with open(val_raw_path, "w", encoding="utf-8") as f:
        json.dump([
            {"prompt_formatted": p, "chosen_formatted": c}
            for p, c in zip(val_raw_simple["prompt_formatted"], val_raw_simple["chosen_formatted"])
        ], f, ensure_ascii=False, indent=2)

    print(f"Validation raw data saved to {val_raw_path}")

    # токенизируем все примеры заранее
    def tokenize_all(examples):
        chosen_enc = tokenizer(
            examples["prompt_formatted"],
            examples["chosen_formatted"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        rejected_enc = tokenizer(
            examples["prompt_formatted"],
            examples["rejected_formatted"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }

    train_tok = train_raw.map(tokenize_all, batched=True)
    val_tok = val_raw.map(tokenize_all, batched=True)

    columns_to_keep = ["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"]
    train_tok = train_tok.remove_columns([col for col in train_tok.column_names if col not in columns_to_keep])
    val_tok = val_tok.remove_columns([col for col in val_tok.column_names if col not in columns_to_keep])

    # преобразуем в списки тензоров
    train_data = {
        "chosen_input_ids": torch.tensor(train_tok["chosen_input_ids"]),
        "chosen_attention_mask": torch.tensor(train_tok["chosen_attention_mask"]),
        "rejected_input_ids": torch.tensor(train_tok["rejected_input_ids"]),
        "rejected_attention_mask": torch.tensor(train_tok["rejected_attention_mask"]),
    }
    val_data = {
        "chosen_input_ids": torch.tensor(val_tok["chosen_input_ids"]),
        "chosen_attention_mask": torch.tensor(val_tok["chosen_attention_mask"]),
        "rejected_input_ids": torch.tensor(val_tok["rejected_input_ids"]),
        "rejected_attention_mask": torch.tensor(val_tok["rejected_attention_mask"]),
    }

    torch.save({"train": train_data, "val": val_data}, DATA_PATH)
    print(f"Dataset tokenized and saved to {DATA_PATH}")

class DPOPairDataset(Dataset):
    def __init__(self, data):
        self.chosen_input_ids = data["chosen_input_ids"]
        self.chosen_attention_mask = data["chosen_attention_mask"]
        self.rejected_input_ids = data["rejected_input_ids"]
        self.rejected_attention_mask = data["rejected_attention_mask"]

    def __len__(self):
        return self.chosen_input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "chosen_input_ids": self.chosen_input_ids[idx],
            "chosen_attention_mask": self.chosen_attention_mask[idx],
            "rejected_input_ids": self.rejected_input_ids[idx],
            "rejected_attention_mask": self.rejected_attention_mask[idx],
        }

def load_prepared_dataset():
    data = torch.load(DATA_PATH)
    train_dataset = DPOPairDataset(data["train"])
    val_dataset = DPOPairDataset(data["val"])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    return train_loader, val_loader

if __name__ == "__main__":
    dataset = load_dataset("Anthropic/hh-rlhf", split="train[:3000]")
    prepare_and_save_dataset(dataset, tokenizer)
