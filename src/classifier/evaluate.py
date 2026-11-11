
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.classifier.dataset import SymptomDataset
from src.classifier.labels import (
    NUM_CLASSES,
    ID_TO_LABEL,
    get_label_id,
)
from src.classifier.model import create_model


DATA_PATH = Path("data/processed/teste.csv")
MODEL_PATH = Path("models/checkpoints/doutorml-distilbert")
REPORT_DIR = Path("reports/classifier")

BATCH_SIZE = 8
MAX_LENGTH = 128


def load_test_data():
    dataframe = pd.read_csv(DATA_PATH)

    if not {"text", "label"}.issubset(dataframe.columns):
        raise ValueError(
            "O arquivo de teste precisa conter as colunas 'text' e 'label'."
        )

    if dataframe[["text", "label"]].isnull().any().any():
        raise ValueError("Foram encontrados valores vazios no conjunto de teste.")

    texts = dataframe["text"].tolist()

    labels = [
        get_label_id(label)
        for label in dataframe["label"].tolist()
    ]

    return texts, labels


def generate_predictions(model, dataloader, device):
    model.eval()

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
            )

            batch_predictions = torch.argmax(outputs.logits, dim=1)

            predictions.extend(batch_predictions.cpu().tolist())
            real_labels.extend(labels.cpu().tolist())

    return real_labels, predictions


def calculate_metrics(real_labels, predictions):
    class_ids = list(range(NUM_CLASSES))

    report = classification_report(
        real_labels,
        predictions,
        labels=class_ids,
        target_names=[
            ID_TO_LABEL[index] for index in class_ids
        ],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(real_labels, predictions),
        "precision_macro": precision_score(
            real_labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            real_labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            real_labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            real_labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics, report


def save_classification_report(report):
    report_dataframe = pd.DataFrame(report).transpose()

    report_dataframe.to_csv(
        REPORT_DIR / "classification_report.csv",
        index=True,
        encoding="utf-8-sig",
    )


def save_confusion_matrix(real_labels, predictions):
    class_ids = list(range(NUM_CLASSES))

    matrix = confusion_matrix(
        real_labels,
        predictions,
        labels=class_ids,
    )

    class_names = [
        ID_TO_LABEL[index] for index in class_ids
    ]

    figure, axis = plt.subplots(figsize=(18, 15))

    image = axis.imshow(matrix, interpolation="nearest")

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=class_ids,
        yticks=class_ids,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Classe prevista",
        ylabel="Classe real",
        title="Matriz de confusão - DoutorML",
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=90,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
    )

    plt.setp(
        axis.get_yticklabels(),
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        REPORT_DIR / "matriz_confusao.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def evaluate():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {MODEL_PATH}. "
            "Execute o treinamento primeiro."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo utilizado: {device}")

    print("\nCarregando conjunto de teste...")

    texts, labels = load_test_data()

    print(f"Quantidade de registros: {len(texts)}")

    print("\nCarregando tokenizer e modelo...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = create_model(str(MODEL_PATH))
    model.to(device)

    test_dataset = SymptomDataset(
        texts,
        labels,
        tokenizer,
        MAX_LENGTH,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("\nGerando previsões...")

    real_labels, predictions = generate_predictions(
        model,
        test_loader,
        device,
    )

    print("\nCalculando métricas...")

    metrics, report = calculate_metrics(
        real_labels,
        predictions,
    )

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    save_classification_report(report)

    save_confusion_matrix(
        real_labels,
        predictions,
    )

    with open(
        REPORT_DIR / "metricas.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nAvaliação concluída!")
    print(f"Resultados salvos em: {REPORT_DIR}")


if __name__ == "__main__":
    evaluate()
