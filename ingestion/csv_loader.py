import pandas as pd
from langchain.schema import Document
from utils.helpers import create_document, chunk_documents


def load_csv(file_path: str) -> list[Document]:
    """
    Reads a CSV file and converts each row into a Document.
    Each row is converted to a readable key:value text format
    so the LLM can understand structured data naturally.
    """
    documents = []
    df = pd.read_csv(file_path)

    # Drop rows where all values are empty
    df.dropna(how="all", inplace=True)

    for row_num, row in df.iterrows():
        # Convert each row to "column: value, column: value" format
        # Example: "Name: Alice, Age: 30, Department: Engineering"
        row_text = ", ".join(
            f"{col}: {val}"
            for col, val in row.items()
            if pd.notna(val)  # skip empty cells
        )

        doc = create_document(
            text=row_text,
            metadata={
                "source": file_path.split("/")[-1],
                "row": row_num + 1,
                "type": "csv"
            }
        )
        documents.append(doc)

    return chunk_documents(documents)