"""
retriever.py - Document Retriever for RAG-Cratoss

Loads the ChromaDB vector store and retrieves the most relevant
document chunks for a given user query using semantic similarity search.
"""

import os
import sys
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Add project root to path so we can import from ingestion
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.embedder import get_embedding_function, CHROMA_PERSIST_DIR, COLLECTION_NAME


# --- Configuration ---
DEFAULT_TOP_K = 5  # Number of chunks to retrieve


class Retriever:
    """
    Semantic search retriever backed by ChromaDB.

    Loads the persisted vector store and supports:
      - Basic similarity search (top-K nearest neighbors)
      - Similarity search with relevance scores
      - Metadata-filtered search (e.g., filter by category)
    """

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.persist_directory = os.path.abspath(persist_directory)
        self.collection_name = collection_name
        self.top_k = top_k

        # Load embedding function (must match what was used during ingestion)
        self.embedding_fn = get_embedding_function()

        # Load the existing vector store
        print(f"📂 Loading vector store from: {self.persist_directory}")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_fn,
            collection_name=self.collection_name,
        )

        doc_count = self.vectorstore._collection.count()
        print(f"✅ Retriever ready! Vector store has {doc_count} document chunks.\n")

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieve the most relevant chunks for a query.

        Uses cosine similarity (since embeddings are normalized) to find
        the top-K nearest document chunks in the vector store.

        Args:
            query:  The user's natural-language question.
            top_k:  Number of results to return (overrides default).

        Returns:
            List of Document objects, ranked by relevance.
        """
        k = top_k or self.top_k
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def retrieve_with_scores(self, query: str, top_k: int = None) -> List[tuple]:
        """
        Retrieve chunks along with their similarity scores.

        Returns (Document, score) tuples. Lower score = more similar
        (ChromaDB returns L2 distance by default; with normalized
        embeddings this correlates inversely with cosine similarity).

        Args:
            query:  The user's natural-language question.
            top_k:  Number of results to return.

        Returns:
            List of (Document, similarity_score) tuples.
        """
        k = top_k or self.top_k
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return results

    def retrieve_with_filter(
        self,
        query: str,
        filter_dict: Dict[str, Any],
        top_k: int = None,
    ) -> List[Document]:
        """
        Retrieve chunks filtered by metadata.

        Useful for scoping retrieval to a specific category, file, or page.
        Example filter: {"category": "protocols"} to only search protocol docs.

        Args:
            query:        The user's question.
            filter_dict:  Metadata filter (e.g., {"category": "hardware"}).
            top_k:        Number of results to return.

        Returns:
            Filtered list of Document objects.
        """
        k = top_k or self.top_k
        results = self.vectorstore.similarity_search(
            query, k=k, filter=filter_dict
        )
        return results

    def get_langchain_retriever(self, top_k: int = None, filter_dict: Dict[str, Any] = None):
        """
        Returns a LangChain-compatible retriever object.

        This can be plugged directly into LangChain chains and pipelines
        (e.g., RetrievalQA, create_retrieval_chain, etc.).

        Args:
            top_k:        Number of results to return.
            filter_dict:  Optional metadata filter.

        Returns:
            A LangChain VectorStoreRetriever instance.
        """
        k = top_k or self.top_k
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)


# ============================================================
# Helper functions to pretty-print results
# ============================================================

def print_results(results: List[Document], query: str):
    """Pretty-print retrieved documents."""
    print(f"{'='*70}")
    print(f"🔍 Query: \"{query}\"")
    print(f"   Found {len(results)} relevant chunks")
    print(f"{'='*70}")

    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        print(f"\n📄 Result {i}")
        print(f"   File:     {meta.get('file_name', 'N/A')}")
        print(f"   Category: {meta.get('category', 'N/A')}")
        print(f"   Page:     {meta.get('page', 'N/A')}")
        print(f"   Chunk #:  {meta.get('chunk_index', 'N/A')}")
        print(f"   {'─'*50}")
        # Show first 300 chars of the chunk content
        content = doc.page_content
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"   {preview}")
    print()


def print_results_with_scores(results: List[tuple], query: str):
    """Pretty-print retrieved documents with their similarity scores."""
    print(f"{'='*70}")
    print(f"🔍 Query: \"{query}\"")
    print(f"   Found {len(results)} relevant chunks (with scores)")
    print(f"{'='*70}")

    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        print(f"\n📄 Result {i}  |  🎯 Relevance Score: {score:.4f}")
        print(f"   File:     {meta.get('file_name', 'N/A')}")
        print(f"   Category: {meta.get('category', 'N/A')}")
        print(f"   Page:     {meta.get('page', 'N/A')}")
        print(f"   Chunk #:  {meta.get('chunk_index', 'N/A')}")
        print(f"   {'─'*50}")
        content = doc.page_content
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"   {preview}")
    print()


# ============================================================
# Run standalone for testing
# ============================================================

if __name__ == "__main__":
    # Initialize the retriever
    retriever = Retriever(top_k=5)

    # --- Test Queries ---
    test_queries = [
        "What is Captial of India?",
        "who is Dharshan Kumar",
        "How much is LOQ Laptop?",
    ]

    for query in test_queries:
        # Test 1: Basic similarity search
        results = retriever.retrieve(query)
        print_results(results, query)

        # Test 2: With relevance scores
        print("\n📊 With Relevance Scores:")
        scored_results = retriever.retrieve_with_scores(query, top_k=3)
        print_results_with_scores(scored_results, query)

        print(f"\n{'#'*70}\n")
