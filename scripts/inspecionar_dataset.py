
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"

csv_path = RAW_DATA_PATH / "Symptom2Disease.csv"

df = pd.read_csv(csv_path)

df = df.drop(columns=["Unnamed: 0"], errors="ignore")

df["label"] = df["label"].str.strip()
df["text"] = df["text"].str.strip()

print("\n=== DISTRIBUIÇÃO DAS DOENÇAS ===")
print(df["label"].value_counts().sort_index().to_string())

print("\n=== ESTATÍSTICAS DAS CLASSES ===")
print(f"Classes: {df['label'].nunique()}")
print(f"Menor quantidade por classe: {df['label'].value_counts().min()}")
print(f"Maior quantidade por classe: {df['label'].value_counts().max()}")

print("\n=== TEXTOS REPETIDOS ===")
repeated_texts = df[df.duplicated(subset=["text"], keep=False)]

print(f"Registros envolvidos em repetições: {len(repeated_texts)}")

if not repeated_texts.empty:
    print(
        repeated_texts.sort_values("text")[
            ["label", "text"]
        ].to_string(index=False)
    )

print("\n=== TEXTOS ASSOCIADOS A MAIS DE UMA DOENÇA ===")
labels_per_text = df.groupby("text")["label"].nunique()
conflicting_texts = labels_per_text[labels_per_text > 1]

print(f"Textos com rótulos conflitantes: {len(conflicting_texts)}")

if not conflicting_texts.empty:
    print(
        df[df["text"].isin(conflicting_texts.index)][
            ["label", "text"]
        ].to_string(index=False)
    )

print("\n=== ANÁLISE CONCLUÍDA ===")
