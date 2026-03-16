"""
loader.py - PDF Document Loader for RAG-Cratoss

Loads all PDF files from data/pdfs/ (including subdirectories like
architecture/, hardware/, protocols/) and returns LangChain Document
objects with metadata (source file path and category).
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# Base path to the PDFs directory (relative to project root)
PDF_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")


def load_pdfs_from_directory(pdf_dir: str = PDF_BASE_DIR) -> List[Document]:
    """
    Recursively loads all PDF files from the given directory.

    Each PDF is loaded page-by-page using PyPDFLoader. Metadata is enriched
    with a 'category' field derived from the subfolder name (e.g., 'hardware',
    'protocols', 'architecture').

    Args:
        pdf_dir: Path to the root directory containing PDF subfolders.

    Returns:
        A list of LangChain Document objects with page_content and metadata.
    """
    pdf_dir = os.path.abspath(pdf_dir)
    all_documents: List[Document] = []

    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    print(f"📂 Scanning for PDFs in: {pdf_dir}")

    for root, _dirs, files in os.walk(pdf_dir):
        for file_name in sorted(files):
            if not file_name.lower().endswith(".pdf"):
                continue

            file_path = os.path.join(root, file_name)

            # Derive category from the subfolder name (e.g., "hardware")
            relative_path = os.path.relpath(file_path, pdf_dir)
            category = relative_path.split(os.sep)[0] if os.sep in relative_path else "general"

            print(f"  📄 Loading: {relative_path} (category: {category})")

            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                # Enrich metadata for each page
                for page in pages:
                    page.metadata["category"] = category
                    page.metadata["file_name"] = file_name

                all_documents.extend(pages)
                print(f"     ✅ Loaded {len(pages)} pages")

            except Exception as e:
                print(f"     ❌ Error loading {file_name}: {e}")

    print(f"\n📊 Total documents loaded: {len(all_documents)} pages from {pdf_dir}")
    return all_documents


# --- Run standalone for testing ---
if __name__ == "__main__":
    docs = load_pdfs_from_directory()
    for doc in docs[:3]:  # Preview first 3 pages
        print(f"\n--- {doc.metadata} ---")
        print(doc.page_content[:300])
