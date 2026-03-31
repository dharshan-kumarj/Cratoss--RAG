"""
reranker.py - Cross-Encoder Reranker for RAG-Cratoss

Takes candidate chunks (e.g., from the hybrid retriever) and reranks
them using a cross-encoder model that scores each (query, document)
pair jointly, producing more accurate relevance estimates than
bi-encoder similarity alone.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (local, no API required)
"""

import torch
from typing import List, Tuple, Dict, Any
from sentence_transformers import CrossEncoder


# --- Configuration ---
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_N = 3


class Reranker:
    """
    Cross-encoder reranker for refining retrieval results.

    Why rerank?
      Bi-encoder retrieval (BM25/dense) is fast but scores query and
      document independently. A cross-encoder processes the (query, doc)
      pair together through a single transformer pass, capturing deeper
      token-level interactions. This typically improves precision at the
      cost of speed — which is fine when reranking a small candidate set.

    Usage:
      reranker = Reranker()
      reranked = reranker.rerank(query, candidate_chunks, top_n=3)
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        """
        Initialize the cross-encoder reranker.

        Automatically selects CUDA if a GPU is available, otherwise
        falls back to CPU.

        Args:
            model_name: HuggingFace model identifier for the cross-encoder.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔄 Loading reranker model: {model_name}")
        print(f"   Device: {self.device}")

        self.model = CrossEncoder(
            model_name,
            device=self.device,
        )

        print(f"✅ Reranker ready!\n")

    def rerank(
        self,
        query: str,
        chunks: List[Tuple[float, str, Dict[str, Any]]],
        top_n: int = DEFAULT_TOP_N,
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """
        Rerank candidate chunks using the cross-encoder.

        Takes the output of a retriever (e.g., HybridRetriever.retrieve_hybrid)
        and rescores each chunk by feeding (query, doc_text) through the
        cross-encoder. Returns the top-N chunks sorted by the new scores.

        Args:
            query:  The user's natural-language question.
            chunks: List of (original_score, doc_text, metadata) tuples
                    from a prior retrieval step.
            top_n:  Number of top results to keep after reranking.

        Returns:
            List of (cross_encoder_score, doc_text, metadata) tuples,
            sorted by descending cross-encoder score, truncated to top_n.
        """
        if not chunks:
            print("⚠️  No chunks to rerank.")
            return []

        input_count = len(chunks)

        # Build (query, doc) pairs for the cross-encoder
        pairs = [(query, doc_text) for _score, doc_text, _meta in chunks]

        # Score all pairs in a single batch
        scores = self.model.predict(pairs)

        # Attach new scores and sort descending
        scored_chunks = [
            (float(score), doc_text, metadata)
            for score, (_orig_score, doc_text, metadata) in zip(scores, chunks)
        ]
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Truncate to top_n
        reranked = scored_chunks[:top_n]

        print(f"🏆 Reranked {input_count} → {len(reranked)} chunks")

        return reranked


# ============================================================
# Run standalone for testing
# ============================================================

if __name__ == "__main__":
    # Quick smoke test with dummy data
    reranker = Reranker()

    dummy_query = "What is MQTT protocol?"
    dummy_chunks = [
        (0.85, "MQTT is a lightweight messaging protocol designed for IoT devices.", {"file_name": "protocols.pdf", "page": 1}),
        (0.72, "CoAP is a web transfer protocol for constrained nodes and networks.", {"file_name": "protocols.pdf", "page": 5}),
        (0.68, "Arduino UNO R3 uses an ATmega328P microcontroller.", {"file_name": "hardware.pdf", "page": 2}),
        (0.55, "MQTT uses a publish-subscribe model with a central broker.", {"file_name": "protocols.pdf", "page": 2}),
        (0.40, "TLS encryption can be applied to MQTT connections for security.", {"file_name": "security.pdf", "page": 10}),
    ]

    print(f"{'='*60}")
    print(f"🔍 Query: \"{dummy_query}\"")
    print(f"   Input: {len(dummy_chunks)} chunks")
    print(f"{'='*60}")

    results = reranker.rerank(dummy_query, dummy_chunks, top_n=3)

    for i, (score, text, meta) in enumerate(results, 1):
        print(f"\n  #{i}  Score: {score:.4f}")
        print(f"       File: {meta.get('file_name', 'N/A')}")
        print(f"       {text}")
    print()
