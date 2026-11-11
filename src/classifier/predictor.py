
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.classifier.labels import NUM_CLASSES, ID_TO_LABEL
from src.classifier.model import create_model
from src.classifier.translations import translate_label


MODEL_PATH = Path("models/checkpoints/doutorml-distilbert")

MAX_LENGTH = 128
DEFAULT_TOP_K = 3


class SymptomPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}. "
                "Execute o treinamento primeiro."
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = create_model(str(model_path))
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str, top_k: int = DEFAULT_TOP_K):
        if not isinstance(text, str):
            raise TypeError("O relato deve ser uma string.")

        if not text.strip():
            raise ValueError("O relato de sintomas não pode estar vazio.")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k deve ser um número inteiro.")

        if not 1 <= top_k <= NUM_CLASSES:
            raise ValueError(
                f"top_k deve estar entre 1 e {NUM_CLASSES}."
            )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=top_k,
            dim=1,
        )

        predictions = []

        for probability, class_id in zip(
            top_probabilities[0],
            top_indices[0],
        ):
            label_id = class_id.item()
            original_label = ID_TO_LABEL[label_id]

            predictions.append({
                "label": translate_label(original_label),
                "probability": round(probability.item(), 4),
            })

        return predictions


if __name__ == "__main__":
    predictor = SymptomPredictor()
    relato = input("Descreva os sintomas: ")
    predictions = predictor.predict(relato)

    print("\nClassificações mais prováveis:\n")

    for index, prediction in enumerate(predictions, start=1):
        print(
            f"{index}. {prediction['label']} "
            f"- {prediction['probability']:.2%}"
        )
