
from typing import Final


CLINICAL_CATEGORIES: Final[tuple[str, ...]] = (
    "respiratorio",
    "gastrointestinal",
    "cardiovascular",
    "neurologico",
    "metabolico",
    "infeccioso",
    "musculoesqueletico",
    "dermatologico",
    "outros",
)


CATEGORY_DESCRIPTIONS: Final[dict[str, str]] = {
    "respiratorio": "Condições relacionadas ao sistema respiratório.",
    "gastrointestinal": "Condições relacionadas ao sistema digestivo.",
    "cardiovascular": "Condições relacionadas ao coração e à circulação.",
    "neurologico": "Condições relacionadas ao sistema nervoso.",
    "metabolico": "Condições relacionadas ao metabolismo.",
    "infeccioso": "Condições associadas a agentes infecciosos.",
    "musculoesqueletico": (
        "Condições relacionadas aos músculos, ossos e articulações."
    ),
    "dermatologico": "Condições relacionadas à pele e seus anexos.",
    "outros": "Condições que não se enquadram nas demais categorias.",
}


LABEL_TO_ID: Final[dict[str, int]] = {
    label: index
    for index, label in enumerate(CLINICAL_CATEGORIES)
}


ID_TO_LABEL: Final[dict[int, str]] = {
    index: label
    for label, index in LABEL_TO_ID.items()
}


NUM_CLASSES: Final[int] = len(CLINICAL_CATEGORIES)


def normalize_label(label: str) -> str:
    """Normaliza uma categoria para comparação."""
    if not isinstance(label, str):
        raise TypeError("A categoria deve ser uma string.")

    return label.strip().lower()


def get_label_id(label: str) -> int:
    """Retorna o identificador numérico de uma categoria."""
    normalized_label = normalize_label(label)

    if normalized_label not in LABEL_TO_ID:
        raise ValueError(
            f"Categoria desconhecida: {label!r}. "
            f"Categorias disponíveis: {', '.join(CLINICAL_CATEGORIES)}"
        )

    return LABEL_TO_ID[normalized_label]


def get_label_name(label_id: int) -> str:
    """Retorna o nome da categoria a partir do identificador."""
    if isinstance(label_id, bool) or not isinstance(label_id, int):
        raise TypeError("O identificador da categoria deve ser um inteiro.")

    if label_id not in ID_TO_LABEL:
        raise ValueError(
            f"Identificador inválido: {label_id}. "
            f"Valores permitidos: 0 a {NUM_CLASSES - 1}."
        )

    return ID_TO_LABEL[label_id]


def get_category_description(label: str) -> str:
    """Retorna a descrição de uma categoria."""
    normalized_label = normalize_label(label)

    if normalized_label not in CATEGORY_DESCRIPTIONS:
        raise ValueError(f"Categoria desconhecida: {label!r}.")

    return CATEGORY_DESCRIPTIONS[normalized_label]


def validate_labels(labels: list[str]) -> None:
    """Verifica se todas as categorias do dataset são válidas."""
    if not labels:
        raise ValueError("A lista de categorias não pode estar vazia.")

    invalid_labels = sorted(
        {
            normalize_label(label)
            for label in labels
            if normalize_label(label) not in LABEL_TO_ID
        }
    )

    if invalid_labels:
        raise ValueError(
            f"Categorias incompatíveis com o modelo: {invalid_labels}"
        )
