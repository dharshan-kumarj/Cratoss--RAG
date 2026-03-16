"""
embedder.py - Embedding & Vector Store for RAG-Cratoss

Takes chunked documents, generates vector embeddings using
HuggingFace sentence-transformers, and stores them in a
ChromaDB persistent vector store for later retrieval.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # Fast, lightweight, 384-dim
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "chroma_db")
COLLECTION_NAME = "rag_cratoss_docs"


def get_embedding_function(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFace embedding function.

    Uses the all-MiniLM-L6-v2 model by default, which produces 384-dimensional
    vectors. This model is a good balance between speed and quality for RAG.

    Args:
        model_name: HuggingFace model identifier for the embedding model.

    Returns:
        A HuggingFaceEmbeddings instance.
    """
    print(f"🤖 Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def store_embeddings(
    chunks: List[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = 50,
) -> Chroma:
    """
    Embeds document chunks and stores them in ChromaDB.

    Processes chunks in batches to avoid memory issues with large document
    collections. The vector store is persisted to disk so it can be reloaded
    without re-embedding.

    Args:
        chunks:            List of chunked Document objects.
        persist_directory: Path to the ChromaDB storage directory.
        collection_name:   Name of the Chroma collection.
        embedding_model:   HuggingFace model name for embeddings.
        batch_size:        Number of chunks to embed at a time.

    Returns:
        A Chroma vector store instance.
    """
    if not chunks:
        print("⚠️  No chunks to embed.")
        return None

    persist_directory = os.path.abspath(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    embedding_fn = get_embedding_function(embedding_model)

    print(f"💾 Storing {len(chunks)} chunks into ChromaDB at: {persist_directory}")
    print(f"   Collection: {collection_name}")

    # Process in batches
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"   📦 Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vectorstore is None:
            # Create the vector store with the first batch
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding_fn,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        else:
            # Add subsequent batches to existing store
            vectorstore.add_documents(documents=batch)

    print(f"✅ Successfully embedded and stored {len(chunks)} chunks!")
    return vectorstore


def load_vectorstore(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> Chroma:
    """
    Loads an existing ChromaDB vector store from disk.

    Use this when you've already run the ingestion pipeline and just
    need to retrieve from the existing embeddings.

    Args:
        persist_directory: Path to the ChromaDB storage directory.
        collection_name:   Name of the Chroma collection.
        embedding_model:   HuggingFace model name (must match what was used to embed).

    Returns:
        A Chroma vector store instance.
    """
    persist_directory = os.path.abspath(persist_directory)
    embedding_fn = get_embedding_function(embedding_model)

    print(f"📂 Loading vector store from: {persist_directory}")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_fn,
        collection_name=collection_name,
    )

    count = vectorstore._collection.count()
    print(f"✅ Loaded vector store with {count} documents.")
    return vectorstore


# --- Run standalone for testing ---
if __name__ == "__main__":
    from ingestion.loader import load_pdfs_from_directory
    from ingestion.chunker import chunk_documents

    # Full pipeline: Load → Chunk → Embed & Store
    docs = load_pdfs_from_directory()
    chunks = chunk_documents(docs)
    vectorstore = store_embeddings(chunks)

    # Quick test: search for something
    if vectorstore:
        query = "What is MQTT protocol?"
        results = vectorstore.similarity_search(query, k=3)
        print(f"\n🔍 Search results for: '{query}'")
        for i, result in enumerate(results):
            print(f"\n  Result {i+1} | {result.metadata.get('file_name')} | Page {result.metadata.get('page')}")
            print(f"  {result.page_content[:200]}...")
