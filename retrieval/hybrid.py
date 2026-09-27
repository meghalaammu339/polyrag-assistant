from langchain.schema import Document
from retrieval.dense_retriever import dense_search
from retrieval.sparse_retriever import build_bm25_index, bm25_search
from vectorstore.store import get_vectorstore


def reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[Document],
    k: int = 60
) -> list[Document]:
    """
    Combines dense and sparse results using Reciprocal Rank Fusion (RRF).
    
    RRF assigns a score to each document based on its rank in each list:
        score = 1 / (k + rank)
    
    Documents appearing in both lists get scores from both — so they
    bubble up to the top. k=60 is the standard constant used in research.
    """
    scores = {}
    doc_map = {}

    # Score dense results
    for rank, doc in enumerate(dense_results):
        key = doc.page_content  # use content as unique key
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    # Score sparse results and add to existing scores
    for rank, doc in enumerate(sparse_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    # Sort by combined score descending
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


def hybrid_search(query: str, collection_name: str = "polyrag", k: int = 5) -> list[Document]:
    """
    Main hybrid search function.
    Runs both dense and sparse search, then fuses results with RRF.
    """
    # Step 1: Dense search from ChromaDB
    dense_results = dense_search(query, collection_name, k=k)

    # Step 2: Get all stored documents to build BM25 index
    vectorstore = get_vectorstore(collection_name)
    all_docs_data = vectorstore.get()

    all_documents = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(
            all_docs_data["documents"],
            all_docs_data["metadatas"]
        )
    ]

    # Step 3: Build BM25 index and run sparse search
    bm25_index = build_bm25_index(all_documents)
    sparse_results = bm25_search(query, all_documents, bm25_index, k=k)

    # Step 4: Fuse both results
    return reciprocal_rank_fusion(dense_results, sparse_results)[:k]