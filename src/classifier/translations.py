
from src.classifier.labels import ID_TO_LABEL


TRANSLATIONS = {
    "Acne": "Acne",
    "Arthritis": "Artrite",
    "Bronchial Asthma": "Asma brônquica",
    "Cervical spondylosis": "Espondilose cervical",
    "Chicken pox": "Catapora",
    "Common Cold": "Resfriado comum",
    "Dengue": "Dengue",
    "Dimorphic Hemorrhoids": "Hemorroidas",
    "Fungal infection": "Infecção fúngica",
    "Hypertension": "Hipertensão",
    "Impetigo": "Impetigo",
    "Jaundice": "Icterícia",
    "Malaria": "Malária",
    "Migraine": "Enxaqueca",
    "Pneumonia": "Pneumonia",
    "Psoriasis": "Psoríase",
    "Typhoid": "Febre tifoide",
    "Varicose Veins": "Varizes",
    "allergy": "Alergia",
    "diabetes": "Diabetes",
    "drug reaction": "Reação medicamentosa",
    "gastroesophageal reflux disease": "Doença do refluxo gastroesofágico",
    "peptic ulcer disease": "Doença ulcerosa péptica",
    "urinary tract infection": "Infecção do trato urinário",
}


def translate_label(label: str) -> str:
    return TRANSLATIONS.get(label, label)


def get_translated_labels() -> dict:
    return {
        class_id: translate_label(label)
        for class_id, label in ID_TO_LABEL.items()
    }
