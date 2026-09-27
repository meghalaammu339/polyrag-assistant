import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Directory where ChromaDB will persist data on disk
CHROMA_DIR = "./chroma_db"

# We use a free HuggingFace embedding model
# "all-MiniLM-L6-v2" is small, fast and works great for general purpose RAG
# No API key needed — runs locally
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Loads and returns the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore(collection_name: str = "polyrag") -> Chroma:
    """
    Connects to an existing ChromaDB collection.
    Used at query time to search stored embeddings.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR
    )


def store_documents(documents: list[Document], collection_name: str = "polyrag"):
    """
    Embeds and stores a list of chunked Documents into ChromaDB.
    If collection already exists, new docs are added to it.
    """
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )
    return vectorstore


def clear_vectorstore(collection_name: str = "polyrag"):
    """
    Deletes all documents from the collection.
    Called when user wants to start a fresh session.
    """
    vectorstore = get_vectorstore(collection_name)
    vectorstore.delete_collection()


def get_all_sources(collection_name: str = "polyrag") -> list[str]:
    """
    Returns a list of all unique sources stored in the vector store.
    Used in the UI to show the user what documents are loaded.
    """
    vectorstore = get_vectorstore(collection_name)
    
    # ChromaDB stores metadata alongside each chunk
    # We extract all metadata and pull unique source names
    data = vectorstore.get()
    sources = set()

    for metadata in data["metadatas"]:
        if metadata and "source" in metadata:
            sources.add(metadata["source"])

    return list(sources)