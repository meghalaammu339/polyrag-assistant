from vectorstore.store import get_vectorstore


def get_dense_retriever(collection_name: str = "polyrag", k: int = 5):
    """
    Returns a LangChain retriever that searches ChromaDB
    using embedding similarity (semantic search).
    k = number of top chunks to retrieve.
    """
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def dense_search(query: str, collection_name: str = "polyrag", k: int = 5):
    """
    Runs a direct semantic similarity search and returns top-k chunks.
    """
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.similarity_search(query, k=k)