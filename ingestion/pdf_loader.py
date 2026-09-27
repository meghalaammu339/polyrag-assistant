import fitz  # PyMuPDF
from langchain.schema import Document
from utils.helpers import create_document, chunk_documents


def load_pdf(file_path: str) -> list[Document]:
    """
    Extracts text from each page of a PDF file,
    wraps each page into a Document with metadata,
    then chunks and returns them.
    """
    documents = []
    pdf = fitz.open(file_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text()

        if not text.strip():
            # Skip empty pages (scanned images, blank pages)
            continue

        doc = create_document(
            text=text,
            metadata={
                "source": file_path.split("/")[-1],  # just the filename
                "page": page_num + 1,                 # 1-indexed for readability
                "type": "pdf"
            }
        )
        documents.append(doc)

    pdf.close()
    return chunk_documents(documents)