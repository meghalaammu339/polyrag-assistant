import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from utils.helpers import create_document, chunk_documents


def load_web(url: str) -> list[Document]:
    """
    Fetches a web page, extracts clean readable text,
    and returns chunked Documents.
    """
    headers = {
        # Some websites block requests without a User-Agent header
        # This makes our request look like it's coming from a real browser
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # raises error if page not found (404 etc.)

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style tags — we only want visible text
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract all visible text
    raw_text = soup.get_text(separator="\n")

    doc = create_document(
        text=raw_text,
        metadata={
            "source": url,
            "type": "web"
        }
    )

    return chunk_documents([doc])