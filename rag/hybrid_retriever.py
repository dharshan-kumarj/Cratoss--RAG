"""
hybrid_retriever.py - Hybrid Retriever for RAG-Cratoss

Combines BM25 keyword search with ChromaDB semantic search using
Reciprocal Rank Fusion (RRF) to produce a single merged ranking.
This gives the best of both worlds: exact keyword matching (BM25)
and deep semantic understanding (dense embeddings).
"""

import os
import sys
from typing import List, Tuple, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.embedder import get_embedding_function, CHROMA_PERSIST_DIR, COLLECTION_NAME


# --- Configuration ---
DEFAULT_TOP_K = 5
RRF_K = 60  # RRF damping constant (standard default)


class HybridRetriever:
    """
    Hybrid retriever that fuses BM25 keyword search and ChromaDB
    semantic search using Reciprocal Rank Fusion (RRF).

    How it works:
      1. At init, loads ALL document chunks from the existing ChromaDB
         collection and builds a BM25 index over their text.
      2. At query time, both BM25 and semantic search produce independent
         ranked lists. RRF merges them into a single ranking by summing
         reciprocal ranks: score(d) = Σ 1/(k + rank_i(d)) for each system i.
      3. The fused results are returned sorted by descending RRF score.

    This approach does NOT require re-ingestion — it reads directly from
    the existing ChromaDB vector store.
    """

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            persist_directory: Path to the ChromaDB persistent storage.
            collection_name:   Name of the ChromaDB collection to load.
        """
        self.persist_directory = os.path.abspath(persist_directory)
        self.collection_name = collection_name

        # --- Load ChromaDB vector store ---
        self.embedding_fn = get_embedding_function()
        print(f"📂 Loading vector store from: {self.persist_directory}")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_fn,
            collection_name=self.collection_name,
        )

        # --- Pull all documents from the collection for BM25 indexing ---
        collection = self.vectorstore._collection
        all_data = collection.get(include=["documents", "metadatas"])

        self.doc_ids = all_data["ids"]
        self.doc_texts = all_data["documents"]       # list of str
        self.doc_metadatas = all_data["metadatas"]    # list of dict

        doc_count = len(self.doc_texts)
        print(f"📚 Loaded {doc_count} chunks from ChromaDB for BM25 indexing.")

        # --- Build BM25 index ---
        # Tokenize using simple whitespace split (lowercased)
        self.tokenized_corpus = [
            doc.lower().split() for doc in self.doc_texts
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"✅ Hybrid retriever ready! (BM25 + semantic, RRF k={RRF_K})\n")

    def _bm25_search(
        self, query: str, top_n: int
    ) -> List[Tuple[float, int]]:
        """
        Run BM25 keyword search over the full corpus.

        Args:
            query: The user's query string.
            top_n: Number of top results to return.

        Returns:
            List of (bm25_score, corpus_index) tuples, sorted descending.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-N indices by score
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(float(scores[i]), int(i)) for i in top_indices]

    def _semantic_search(
        self, query: str, top_n: int
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """
        Run semantic similarity search via ChromaDB.

        Args:
            query: The user's query string.
            top_n: Number of top results to return.

        Returns:
            List of (relevance_score, doc_text, metadata) tuples.
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=top_n
        )
        return [
            (float(score), doc.page_content, doc.metadata)
            for doc, score in results
        ]

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[str]],
        k: int = RRF_K,
    ) -> Dict[str, float]:
        """
        Compute Reciprocal Rank Fusion scores across multiple ranked lists.

        RRF score for document d = Σ  1 / (k + rank_i(d))
        where rank_i(d) is the 1-based rank of d in list i.

        Args:
            ranked_lists: List of ranked document-ID lists (each ordered
                          best-first).
            k:            RRF damping constant (default 60).

        Returns:
            Dict mapping doc_id -> fused RRF score.
        """
        fused_scores: Dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list, start=1):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return fused_scores

    def retrieve_hybrid(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """
        Retrieve documents using hybrid BM25 + semantic search with RRF fusion.

        Steps:
          1. Run BM25 over the full corpus → ranked list by keyword relevance
          2. Run semantic search via ChromaDB → ranked list by embedding similarity
          3. Merge both lists using Reciprocal Rank Fusion (k=60)
          4. Return the top-K documents sorted by fused score

        Args:
            query: The user's natural-language question.
            top_k: Number of results to return after fusion.

        Returns:
            List of (fused_score, document_text, metadata) tuples,
            sorted by descending fused score, truncated to top_k.
        """
        # Cast a wider net for each sub-retriever before fusion
        fetch_n = top_k * 3

        # --- BM25 search ---
        bm25_results = self._bm25_search(query, top_n=fetch_n)

        # --- Semantic search ---
        semantic_results = self._semantic_search(query, top_n=fetch_n)

        # --- Build lookup tables keyed by ChromaDB doc ID ---
        # BM25 results: map corpus index → doc_id
        bm25_ranked_ids = []
        doc_store: Dict[str, Tuple[str, Dict[str, Any]]] = {}

        for _score, idx in bm25_results:
            doc_id = self.doc_ids[idx]
            bm25_ranked_ids.append(doc_id)
            doc_store[doc_id] = (self.doc_texts[idx], self.doc_metadatas[idx])

        # Semantic results: identify by matching text content to doc_id
        # Build a reverse-lookup from text → doc_id for matching
        text_to_id: Dict[str, str] = {
            text: did for did, text in zip(self.doc_ids, self.doc_texts)
        }

        semantic_ranked_ids = []
        for _score, text, metadata in semantic_results:
            doc_id = text_to_id.get(text, f"sem_{hash(text)}")
            semantic_ranked_ids.append(doc_id)
            doc_store[doc_id] = (text, metadata)

        # --- Reciprocal Rank Fusion ---
        fused_scores = self._reciprocal_rank_fusion(
            [bm25_ranked_ids, semantic_ranked_ids], k=RRF_K
        )

        # Sort by fused score descending
        sorted_ids = sorted(fused_scores.keys(), key=lambda d: fused_scores[d], reverse=True)

        # Build final output
        results: List[Tuple[float, str, Dict[str, Any]]] = []
        for doc_id in sorted_ids[:top_k]:
            score = fused_scores[doc_id]
            text, metadata = doc_store[doc_id]
            results.append((score, text, metadata))

        print(
            f"🔀 Hybrid retrieval: BM25({len(bm25_ranked_ids)}) + "
            f"Semantic({len(semantic_ranked_ids)}) → {len(results)} fused results"
        )
        return results


# ============================================================
# Run standalone for testing
# ============================================================

if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "What is MQTT protocol and how does it work?",
        "Arduino UNO R3 specifications",
        "IoT security vulnerabilities and threats",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"🔍 Query: \"{query}\"")
        print(f"{'='*70}")

        results = retriever.retrieve_hybrid(query, top_k=5)
        for i, (score, text, meta) in enumerate(results, 1):
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"\n  #{i}  Score: {score:.4f}")
            print(f"       File: {meta.get('file_name', 'N/A')}")
            print(f"       Category: {meta.get('category', 'N/A')}")
            print(f"       {preview}")
        print()
