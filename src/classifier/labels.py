from typing import Final


DISEASE_LABELS: Final[tuple[str, ...]] = (
    "Acne",
    "Arthritis",
    "Bronchial Asthma",
    "Cervical spondylosis",
    "Chicken pox",
    "Common Cold",
    "Dengue",
    "Dimorphic Hemorrhoids",
    "Fungal infection",
    "Hypertension",
    "Impetigo",
    "Jaundice",
    "Malaria",
    "Migraine",
    "Pneumonia",
    "Psoriasis",
    "Typhoid",
    "Varicose Veins",
    "allergy",
    "diabetes",
    "drug reaction",
    "gastroesophageal reflux disease",
    "peptic ulcer disease",
    "urinary tract infection",
)


LABEL_TO_ID: Final[dict[str, int]] = {
    label: index
    for index, label in enumerate(DISEASE_LABELS)
}


ID_TO_LABEL: Final[dict[int, str]] = {
    index: label
    for label, index in LABEL_TO_ID.items()
}


NUM_CLASSES: Final[int] = len(DISEASE_LABELS)


def normalize_label(label: str) -> str:
    """Normaliza o rótulo para comparação."""
    if not isinstance(label, str):
        raise TypeError("O rótulo deve ser uma string.")

    return label.strip().casefold()


def get_label_id(label: str) -> int:
    """Retorna o ID numérico correspondente à doença."""
    normalized_label = normalize_label(label)

    for disease, label_id in LABEL_TO_ID.items():
        if normalize_label(disease) == normalized_label:
            return label_id

    raise ValueError(
        f"Doença desconhecida: {label!r}. "
        f"Classes disponíveis: {', '.join(DISEASE_LABELS)}"
    )


def get_label_name(label_id: int) -> str:
    """Retorna o nome da doença correspondente ao ID."""
    if isinstance(label_id, bool) or not isinstance(label_id, int):
        raise TypeError("O ID da doença deve ser um número inteiro.")

    if label_id not in ID_TO_LABEL:
        raise ValueError(
            f"ID inválido: {label_id}. "
            f"Valores permitidos: 0 a {NUM_CLASSES - 1}."
        )

    return ID_TO_LABEL[label_id]


def validate_labels(labels: list[str]) -> None:
    """Verifica se todos os rótulos pertencem às classes do dataset."""
    if not labels:
        raise ValueError("A lista de rótulos não pode estar vazia.")

    valid_labels = {
        normalize_label(label)
        for label in DISEASE_LABELS
    }

    invalid_labels = sorted(
        {
            normalize_label(label)
            for label in labels
            if normalize_label(label) not in valid_labels
        }
    )

    if invalid_labels:
        raise ValueError(
            f"Rótulos incompatíveis com o modelo: {invalid_labels}"
        )