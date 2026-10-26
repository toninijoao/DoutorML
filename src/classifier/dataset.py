from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"


class SymptomDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def carregar_dados(nome_arquivo: str) -> pd.DataFrame:
    file_path = PROCESSED_DATA_PATH / nome_arquivo

    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {file_path}. "
            "Execute primeiro o preprocessing.py."
        )

    df = pd.read_csv(file_path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(
            f"O arquivo {nome_arquivo} precisa conter as colunas 'text' e 'label'."
        )

    return df


def carregar_datasets():
    treino = carregar_dados("treino.csv")
    validacao = carregar_dados("validacao.csv")
    teste = carregar_dados("teste.csv")

    return treino, validacao, teste