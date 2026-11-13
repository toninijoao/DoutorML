
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


KNOWLEDGE_BASE_PATH = Path("data/knowledge_base")


def load_documents(
    directory: str | Path = KNOWLEDGE_BASE_PATH,
) -> list[Document]:
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Diretório da base de conhecimento não encontrado: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"O caminho informado não é um diretório: {directory}"
        )

    pdf_files = sorted(directory.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Nenhum arquivo PDF encontrado em: {directory}"
        )

    documents = []

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded_documents = loader.load()

            documents.extend(loaded_documents)

        except Exception as error:
            raise RuntimeError(
                f"Erro ao carregar o documento {pdf_file}: {error}"
            ) from error

    return documents
