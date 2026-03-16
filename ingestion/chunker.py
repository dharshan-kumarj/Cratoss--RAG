"""
chunker.py - Text Chunking for RAG-Cratoss

Takes the loaded Document objects (full pages) and splits them into
smaller, overlapping chunks suitable for embedding and retrieval.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Chunking Configuration ---
CHUNK_SIZE = 1000       # Max characters per chunk
CHUNK_OVERLAP = 200     # Overlap between consecutive chunks (for context continuity)

# Separators ordered by priority — splits on paragraphs first, then
# sentences, then words, and finally characters as a last resort.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks.

    Uses RecursiveCharacterTextSplitter which tries to keep paragraphs
    and sentences together before falling back to arbitrary character splits.
    All original metadata (source, page, category, file_name) is preserved
    in each chunk.

    Args:
        documents:     List of Document objects from the loader.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of smaller Document objects (chunks) with preserved metadata.
    """
    if not documents:
        print("⚠️  No documents to chunk.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    print(f"✂️  Chunking {len(documents)} documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")

    chunks = text_splitter.split_documents(documents)

    # Add a chunk_index to each chunk's metadata for traceability
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    print(f"✅ Created {len(chunks)} chunks from {len(documents)} pages.")
    return chunks


# --- Run standalone for testing ---
if __name__ == "__main__":
    from ingestion.loader import load_pdfs_from_directory

    docs = load_pdfs_from_directory()
    chunks = chunk_documents(docs)

    # Preview a few chunks
    for chunk in chunks[:3]:
        print(f"\n--- Chunk {chunk.metadata.get('chunk_index')} | {chunk.metadata.get('file_name')} | Page {chunk.metadata.get('page')} ---")
        print(chunk.page_content[:200])
        print(f"... ({len(chunk.page_content)} chars)")
