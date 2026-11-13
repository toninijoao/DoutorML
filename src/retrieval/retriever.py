
from langchain_core.documents import Document

from src.retrieval.vectorstore import get_vectorstore


DEFAULT_TOP_K = 5


def create_retriever(top_k: int = DEFAULT_TOP_K):
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k deve ser um número inteiro.")

    if top_k <= 0:
        raise ValueError("top_k deve ser maior que zero.")

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def retrieve_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[Document]:
    if not isinstance(query, str):
        raise TypeError("A consulta deve ser uma string.")

    if not query.strip():
        raise ValueError("A consulta não pode estar vazia.")

    retriever = create_retriever(top_k=top_k)

    return retriever.invoke(query)
