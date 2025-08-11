import os
import json
import evaluate
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# пути и параметры
save_path = "dpo_finetuned_gpt2"
losses_path = "losses.json"

base_model_name = "gpt2"

num_eval_samples = 10
max_new_tokens = 80

# Загрузка сохранённой модели и токенайзера
model = AutoModelForCausalLM.from_pretrained(save_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(save_path)
model.eval()

# Загрузка базовой модели для сравнения
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
base_model.eval()

val_raw_path = os.path.join(os.path.dirname(__file__), "val_raw.json")

with open(val_raw_path, "r", encoding="utf-8") as f:
    val_data_raw = json.load(f)
# Загрузка логов обучения для графиков
with open(losses_path, "r") as f:
    logs = json.load(f)

train_losses = logs.get("train_losses", [])
val_losses = logs.get("val_losses", [])
train_accuracies = logs.get("train_accuracies", [])  # если есть в json, иначе []
val_accuracies = logs.get("val_accuracies", [])

# Рисуем графики
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss
axes[0].plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
axes[0].plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("DPO Training Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy (если есть)
if train_accuracies and val_accuracies:
    axes[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
    axes[1].plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("DPO Training Accuracy")
    axes[1].legend()
    axes[1].grid(True)
else:
    axes[1].text(0.5, 0.5, "Accuracy data not available", ha='center', va='center')
    axes[1].axis('off')

plt.tight_layout()
plt.show()

# BERTScore
bertscore = evaluate.load("bertscore")

def generate_response(model, tokenizer, prompt, max_new_tokens=max_new_tokens):
    prompt = prompt.strip()
    if not prompt.endswith("Assistant:"):
        prompt += "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=8,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
    return answer

# Выбираем случайные подвыборки для оценки
val_samples = random.sample(val_data_raw, min(num_eval_samples, len(val_data_raw)))

references, base_outputs, tuned_outputs = [], [], []

for idx, sample in enumerate(val_samples):
    prompt = sample['prompt_formatted']
    ref_answer = sample['chosen_formatted']

    tuned_out = generate_response(model, tokenizer, prompt)
    base_out = generate_response(base_model, base_tokenizer, prompt)

    print(f"\n=== Prompt {idx + 1} ===\n{prompt}\n{'-' * 40}")
    print(f"Reference:\n{ref_answer}\n{'-' * 40}")
    print(f"Base model:\n{base_out}\n{'-' * 40}")
    print(f"Tuned model:\n{tuned_out}\n{'=' * 40}")

    references.append([ref_answer])
    base_outputs.append(base_out)
    tuned_outputs.append(tuned_out)

# Подсчёт BERTScore для тонкой модели
bert_results = bertscore.compute(
    predictions=tuned_outputs,
    references=[r[0] for r in references],
    lang="en"
)
mean_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])
print(f"\nTuned BERTScore (mean F1): {mean_f1:.4f}")

# Сохраняем результаты
results = []
for i in range(len(val_samples)):
    results.append({
        "prompt": val_samples[i]['prompt_formatted'],
        "reference": references[i][0],
        "base_output": base_outputs[i],
        "tuned_output": tuned_outputs[i]
    })

os.makedirs("results", exist_ok=True)
with open("results/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Результаты сохранены в results/eval_results.json")
