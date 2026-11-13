
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.configuracoes import EMBEDDING_MODEL


def create_embedding_model() -> HuggingFaceEmbeddings:
    if not EMBEDDING_MODEL:
        raise ValueError(
            "O modelo de embeddings não foi definido nas configurações."
        )

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 16,
        },
        query_encode_kwargs={
            "normalize_embeddings": True,
        },
    )
