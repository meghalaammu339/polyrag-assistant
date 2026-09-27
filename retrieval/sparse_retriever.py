from rank_bm25 import BM25Okapi
from langchain.schema import Document


def build_bm25_index(documents: list[Document]):
    """
    Builds a BM25 index from a list of Documents.
    BM25 works on tokenized words, so we split each chunk by spaces.
    """
    tokenized_corpus = [
        doc.page_content.lower().split() for doc in documents
    ]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def bm25_search(query: str, documents: list[Document], bm25, k: int = 5) -> list[Document]:
    """
    Scores all documents against the query using BM25
    and returns top-k results.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair each document with its BM25 score, sort descending
    scored_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in scored_docs[:k]]