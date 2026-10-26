
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Symptom2Disease.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42


def carregar_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df[["text", "label"]].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df.dropna(subset=["text", "label"])
    df = df[(df["text"] != "") & (df["label"] != "")]

    df = df.drop_duplicates(subset=["text"], keep="first")
    df = df.reset_index(drop=True)

    return df


def dividir_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    treino, temporario = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    validacao, teste = train_test_split(
        temporario,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporario["label"],
    )

    return treino, validacao, teste


def preparar_dataset() -> None:
    df = carregar_dataset()
    treino, validacao, teste = dividir_dataset(df)

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    treino.to_csv(
        PROCESSED_DATA_PATH / "treino.csv",
        index=False,
    )
    validacao.to_csv(
        PROCESSED_DATA_PATH / "validacao.csv",
        index=False,
    )
    teste.to_csv(
        PROCESSED_DATA_PATH / "teste.csv",
        index=False,
    )

    print(f"Total após limpeza: {len(df)}")
    print(f"Treino: {len(treino)}")
    print(f"Validação: {len(validacao)}")
    print(f"Teste: {len(teste)}")
    print(f"Classes: {df['label'].nunique()}")
    print(f"Arquivos salvos em: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preparar_dataset()
