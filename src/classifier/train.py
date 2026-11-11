
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.classifier.dataset import SymptomDataset
from src.classifier.labels import get_label_id
from src.classifier.model import MODEL_NAME, create_model


RANDOM_SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("models/checkpoints/doutorml-distilbert")


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    train_df = pd.read_csv(DATA_DIR / "treino.csv")
    validation_df = pd.read_csv(DATA_DIR / "validacao.csv")

    for dataframe in (train_df, validation_df):
        if not {"text", "label"}.issubset(dataframe.columns):
            raise ValueError(
                "Os arquivos CSV precisam conter as colunas 'text' e 'label'."
            )

        if dataframe[["text", "label"]].isnull().any().any():
            raise ValueError("Foram encontrados valores vazios nos dados.")

    train_texts = train_df["text"].tolist()
    validation_texts = validation_df["text"].tolist()

    train_labels = [
        get_label_id(label) for label in train_df["label"].tolist()
    ]

    validation_labels = [
        get_label_id(label) for label in validation_df["label"].tolist()
    ]

    return (
        train_texts,
        train_labels,
        validation_texts,
        validation_labels,
    )


def evaluate_model(model, dataloader, device):
    model.eval()

    total_loss = 0
    predictions = []
    real_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            batch_predictions = torch.argmax(outputs.logits, dim=1)

            predictions.extend(batch_predictions.cpu().tolist())
            real_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(real_labels, predictions)
    macro_f1 = f1_score(
        real_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1


def train():
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo utilizado: {device}")

    (
        train_texts,
        train_labels,
        validation_texts,
        validation_labels,
    ) = load_data()

    print(f"Registros de treinamento: {len(train_texts)}")
    print(f"Registros de validação: {len(validation_texts)}")

    print("\nCarregando tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SymptomDataset(
        train_texts,
        train_labels,
        tokenizer,
        MAX_LENGTH,
    )

    validation_dataset = SymptomDataset(
        validation_texts,
        validation_labels,
        tokenizer,
        MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("\nCarregando modelo...")

    model = create_model()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0

    print("\nIniciando treinamento...\n")

    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)

        validation_loss, accuracy, macro_f1 = evaluate_model(
            model,
            validation_loader,
            device,
        )

        print(f"Época {epoch + 1}/{EPOCHS}")
        print(f"Loss de treinamento: {average_train_loss:.4f}")
        print(f"Loss de validação: {validation_loss:.4f}")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print("-" * 40)

        if macro_f1 > best_f1:
            best_f1 = macro_f1

            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)

            print("Melhor modelo salvo!")

    print("\nTreinamento concluído!")
    print(f"Melhor Macro F1: {best_f1:.4f}")
    print(f"Modelo salvo em: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()
