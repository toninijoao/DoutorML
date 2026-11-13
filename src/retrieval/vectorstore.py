
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config.configuracoes import (
    VECTORSTORE_NAME,
    VECTORSTORE_PATH,
)
from src.embeddings.embedding_model import create_embedding_model


def get_vectorstore() -> Chroma:
    if not VECTORSTORE_NAME:
        raise ValueError("O nome do vectorstore não foi configurado.")

    if not VECTORSTORE_PATH:
        raise ValueError("O caminho do vectorstore não foi configurado.")

    embedding_model = create_embedding_model()

    return Chroma(
        collection_name=VECTORSTORE_NAME,
        embedding_function=embedding_model,
        persist_directory=str(Path(VECTORSTORE_PATH)),
    )


def add_documents(
    documents: list[Document],
    vectorstore: Chroma | None = None,
) -> Chroma:
    if not documents:
        raise ValueError("A lista de documentos não pode estar vazia.")

    if not all(isinstance(document, Document) for document in documents):
        raise TypeError("Todos os itens devem ser objetos Document.")

    vectorstore = vectorstore or get_vectorstore()

    document_ids = []

    for document in documents:
        identifier = hashlib.sha256(
            (
                document.metadata.get("source", "")
                + str(document.metadata.get("page", ""))
                + document.page_content
            ).encode("utf-8")
        ).hexdigest()

        document_ids.append(identifier)

    vectorstore.add_documents(
        documents=documents,
        ids=document_ids,
    )

    return vectorstore
