import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def clean_text(text: str) -> str:
    """Cleans raw text by removing noise, extra spaces and newlines."""
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def create_document(text: str, metadata: dict) -> Document:
    """Wraps cleaned text and metadata into a LangChain Document."""
    return Document(page_content=clean_text(text), metadata=metadata)


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits documents into smaller overlapping chunks.
    Uses RecursiveCharacterTextSplitter which splits at natural 
    boundaries — paragraph → sentence → word → character.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


def format_source_label(metadata: dict) -> str:
    """Builds a human-readable citation label from chunk metadata."""
    source_type = metadata.get("type", "unknown")
    source = metadata.get("source", "Unknown Source")

    if source_type == "pdf":
        return f"📄 PDF: {source} (Page {metadata.get('page', '?')})"
    elif source_type == "csv":
        return f"📊 CSV: {source} (Row {metadata.get('row', '?')})"
    elif source_type == "web":
        return f"🌐 Web: {source}"
    elif source_type == "youtube":
        return f"🎥 YouTube: {source}"
    else:
        return f"📁 Source: {source}"