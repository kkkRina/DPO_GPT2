import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Гиперпараметры
BATCH_SIZE = 8
NUM_EPOCHS = 12
BETA = 0.2
PATIENCE = 2

losses_path = "../results/losses.json"
save_path = "../results/dpo_finetuned_gpt2"

# загрузка модели
model_name = "gpt2"
# model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# датасет
class DPOPairDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data["chosen_input_ids"])

    def __getitem__(self, idx):
        return {
            "chosen_input_ids": self.data["chosen_input_ids"][idx],
            "chosen_attention_mask": self.data["chosen_attention_mask"][idx],
            "rejected_input_ids": self.data["rejected_input_ids"][idx],
            "rejected_attention_mask": self.data["rejected_attention_mask"][idx],
        }

# загрузка токенизированных данных
tokenized_dataset = torch.load("../data/prepared_pairs.pt")

train_dataset = DPOPairDataset(tokenized_dataset["train"])
val_dataset = DPOPairDataset(tokenized_dataset["val"])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def calculate_sequence_log_probs(model, input_ids, attention_mask):
    """
    Вычисляет среднее логарифмическое значение вероятности токенов в последовательности с учётом маски.
    :param model: модель
    :param input_ids: tensor [batch, seq_len]
    :param attention_mask: tensor [batch, seq_len]
    :return: tensor [batch] среднее log-probs на токен
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    selected_log_probs = torch.gather(
        log_probs, dim=2, index=input_ids.unsqueeze(-1)
    ).squeeze(-1)

    seq_log_probs = (selected_log_probs * attention_mask).sum(dim=1)
    lengths = attention_mask.sum(dim=1).clamp(min=1)
    avg_log_probs = seq_log_probs / lengths
    return avg_log_probs

def DPO(dataloader, model, optimizer, beta=BETA):
    """
    Тренировочный шаг DPO.
    :param dataloader: DataLoader
    :param model: модель
    :param optimizer: оптимизатор
    :param beta: коэффициент температуры
    :return: средний loss по батчам
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_attention_mask = batch["rejected_attention_mask"]

        sequence_log_probs_ch = calculate_sequence_log_probs(
            model, chosen_input_ids, chosen_attention_mask
        )
        sequence_log_probs_rej = calculate_sequence_log_probs(
            model, rejected_input_ids, rejected_attention_mask
        )

        diff = sequence_log_probs_ch - sequence_log_probs_rej
        target = torch.ones_like(diff)

        loss = F.binary_cross_entropy_with_logits(diff * beta, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_dpo(dataloader, model, beta=BETA):
    """
    Валидация DPO без градиентов.
    :param dataloader: DataLoader
    :param model: модель
    :param beta: коэффициент температуры
    :return: (средний loss, accuracy)
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            chosen_input_ids = batch["chosen_input_ids"]
            chosen_attention_mask = batch["chosen_attention_mask"]
            rejected_input_ids = batch["rejected_input_ids"]
            rejected_attention_mask = batch["rejected_attention_mask"]

            sequence_log_probs_ch = calculate_sequence_log_probs(
                model, chosen_input_ids, chosen_attention_mask
            )
            sequence_log_probs_rej = calculate_sequence_log_probs(
                model, rejected_input_ids, rejected_attention_mask
            )

            diff = sequence_log_probs_ch - sequence_log_probs_rej
            target = torch.ones_like(diff)

            loss = F.binary_cross_entropy_with_logits(diff * beta, target)

            total_loss += loss.item()
            total_correct += (sequence_log_probs_ch > sequence_log_probs_rej).sum().item()
            total_samples += chosen_input_ids.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def train_dpo(dataloader, val_dataloader, model, optimizer, num_epochs=NUM_EPOCHS, beta=BETA, patience=PATIENCE,
              save_path=save_path, losses_path=losses_path):
    """
    Основной цикл тренировки с ранней остановкой.
    Сохраняет модель и логи.
    :param dataloader: DataLoader для тренировки
    :param val_dataloader: DataLoader для валидации
    :param model: модель
    :param optimizer: оптимизатор
    :param num_epochs: число эпох
    :param beta: коэффициент температуры
    :param patience: терпение для ранней остановки
    :param save_path: путь для сохранения модели
    :param losses_path: путь для сохранения лоссов и метрик
    """
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss = DPO(dataloader, model, optimizer, beta)
        val_loss, val_acc = validate_dpo(val_dataloader, model, beta)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Сохраняем логи
        os.makedirs(os.path.dirname(losses_path), exist_ok=True)
        with open(losses_path, "w") as f:
            json.dump({
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies
            }, f)

        # сохраняем модель при улучшении
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"New best model saved at epoch {epoch}")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    print("Training finished")


if __name__ == "__main__":
    train_dpo(train_loader, val_loader, model, optimizer)