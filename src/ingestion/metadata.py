
from pathlib import Path

from langchain_core.documents import Document


def enrich_metadata(documents: list[Document]) -> list[Document]:
    if not documents:
        raise ValueError("A lista de documentos não pode estar vazia.")

    enriched_documents = []

    for document in documents:
        metadata = document.metadata.copy()

        source = metadata.get("source")

        if source:
            source_path = Path(source)
            metadata["source"] = source_path.name
            metadata["source_path"] = str(source_path)

        metadata["page"] = metadata.get("page", None)
        metadata["document_type"] = "medical_reference"

        enriched_documents.append(
            Document(
                page_content=document.page_content,
                metadata=metadata,
            )
        )

    return enriched_documents
