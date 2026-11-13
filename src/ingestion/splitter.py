
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    if not documents:
        raise ValueError("A lista de documentos não pode estar vazia.")

    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap não pode ser negativo.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap deve ser menor que chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Nenhum chunk foi gerado.")

    return chunks
