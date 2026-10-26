import pandas as pd
import torch

from src.classifier.dataset import (
    SymptomDataset,
    carregar_dados,
    carregar_datasets,
)


def testar_carregamento():
    treino, validacao, teste = carregar_datasets()

    print("\n=== CARREGAMENTO DOS DADOS ===")
    print(f"Treino: {len(treino)} registros")
    print(f"Validação: {len(validacao)} registros")
    print(f"Teste: {len(teste)} registros")

    assert len(treino) > 0
    assert len(validacao) > 0
    assert len(teste) > 0

    for nome, df in [
        ("Treino", treino),
        ("Validação", validacao),
        ("Teste", teste),
    ]:
        assert {"text", "label"}.issubset(df.columns)
        assert df["text"].notna().all()
        assert df["label"].notna().all()
        print(f"{nome}: estrutura válida")


def testar_dataset():
    texts = [
        "I have a headache and sensitivity to light.",
        "I have a skin rash and itching.",
    ]

    labels = [0, 1]

    class TokenizerFake:
        def __call__(
            self,
            text,
            truncation,
            padding,
            max_length,
            return_tensors,
        ):
            return {
                "input_ids": torch.zeros((1, max_length), dtype=torch.long),
                "attention_mask": torch.ones((1, max_length), dtype=torch.long),
            }

    dataset = SymptomDataset(
        texts=texts,
        labels=labels,
        tokenizer=TokenizerFake(),
        max_length=16,
    )

    item = dataset[0]

    print("\n=== TESTE DA CLASSE SymptomDataset ===")
    print(f"Quantidade de exemplos: {len(dataset)}")
    print(f"Formato de input_ids: {item['input_ids'].shape}")
    print(f"Formato de attention_mask: {item['attention_mask'].shape}")
    print(f"Rótulo: {item['labels'].item()}")

    assert len(dataset) == 2
    assert item["input_ids"].shape == (16,)
    assert item["attention_mask"].shape == (16,)
    assert item["labels"].item() == 0

    print("SymptomDataset funcionando corretamente!")


if __name__ == "__main__":
    testar_carregamento()
    testar_dataset()