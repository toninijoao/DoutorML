from transformers import AutoModelForSequenceClassification

from src.classifier.labels import NUM_CLASSES, ID_TO_LABEL, LABEL_TO_ID


MODEL_NAME = "distilbert-base-uncased"


def create_model(model_name: str = MODEL_NAME):
    """
    Carrega um modelo DistilBERT pré-treinado adaptado
    para classificação das doenças do dataset.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        problem_type="single_label_classification",
    )

    return model
