"""
pipeline.py - RAG Generation Pipeline for RAG-Cratoss

Takes a user query, retrieves relevant context using hybrid retrieval
(BM25 + semantic search with RRF fusion), reranks with a cross-encoder,
applies a 3-tier confidence check, and generates a grounded answer
using Meta's Llama 3.2 running locally via Ollama.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.hybrid_retriever import HybridRetriever
from rag.reranker import Reranker


# --- Configuration ---
LLM_MODEL = "llama3.2"          # Llama 3.2 3B (runs fast on GPU)
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
RELEVANCE_THRESHOLD = 0.25    # Chunks below this score are discarded
LOW_CONFIDENCE_THRESHOLD = 0.45  # Below this: low-confidence warning
TOP_K = 5                     # Number of chunks to retrieve
RERAN_TOP_N = 3               # Chunks kept after reranking


def get_confidence_tier(max_score: float) -> str:
    """
    Determine the confidence tier based on the top reranked chunk score.

    Tiers:
      - "none"  (max_score < 0.25):  No relevant context found.
      - "low"   (0.25 <= max_score < 0.45): Weakly matched context.
      - "full"  (max_score >= 0.45): Strong match, full answer.

    Args:
        max_score: The highest relevance/reranker score among candidate chunks.

    Returns:
        One of "none", "low", or "full".
    """
    if max_score < RELEVANCE_THRESHOLD:
        return "none"
    elif max_score < LOW_CONFIDENCE_THRESHOLD:
        return "low"
    else:
        return "full"


# --- Prompt Template ---
RAG_PROMPT_TEMPLATE = """You are an expert assistant specializing in IoT (Internet of Things), 
network protocols, embedded hardware, and cybersecurity.

Answer the user's question based ONLY on the context provided below. 
Follow these rules strictly:
1. Use ONLY the information from the context to answer.
2. If the context does not contain enough information to answer the question, 
   say: "I don't have enough information about this in my documents."
3. Be specific and cite which document the information comes from when possible.
4. Keep your answer clear, concise, and well-structured.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    question: str
    answer: str
    source_documents: List[Document]
    relevance_scores: List[float]
    has_relevant_context: bool


class RAGPipeline:
    """
    Full RAG pipeline: Query → Hybrid Retrieve → Rerank → Confidence Check → Generate.

    Uses:
      - HybridRetriever (BM25 + semantic search with RRF fusion)
      - Cross-encoder reranker for precision reranking
      - 3-tier confidence system (none / low / full)
      - Llama 3.2 (running locally via Ollama) for answer generation
    """

    def __init__(
        self,
        llm_model: str = LLM_MODEL,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
        top_k: int = TOP_K,
        rerank_top_n: int = RERAN_TOP_N,
    ):
        self.relevance_threshold = relevance_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n

        # --- Initialize Hybrid Retriever + Reranker ---
        print("=" * 60)
        print("🔧 Initializing RAG Pipeline...")
        print("=" * 60)
        self.retriever = HybridRetriever()
        self.reranker = Reranker()

        # --- Initialize LLM (Local via Ollama) ---
        print(f"🤖 Loading LLM: {llm_model} (local via Ollama)")
        print(f"   Max tokens: {max_new_tokens} | Temperature: {temperature}")
        print(f"   Relevance threshold: {relevance_threshold}")
        print(f"   Low-confidence threshold: {low_confidence_threshold}")

        self.llm = ChatOllama(
            model=llm_model,
            temperature=temperature,
            num_predict=max_new_tokens,
        )

        # --- Build Prompt ---
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=RAG_PROMPT_TEMPLATE,
        )

        # --- Build Chain ---
        self.chain = self.prompt | self.llm | StrOutputParser()

        print("✅ RAG Pipeline ready!\n")

    def _format_context(
        self, chunks: List[tuple]
    ) -> str:
        """
        Formats reranked chunks into a context string for the prompt.

        Each chunk is labeled with its source file, category, and page number
        so the LLM can cite sources in its answer.

        Args:
            chunks: List of (score, doc_text, metadata) tuples.
        """
        if not chunks:
            return "No relevant documents found."

        context_parts = []
        for i, (score, text, meta) in enumerate(chunks, 1):
            source_info = (
                f"[Source {i}: {meta.get('file_name', 'Unknown')} | "
                f"Category: {meta.get('category', 'N/A')} | "
                f"Page: {meta.get('page', 'N/A')}]"
            )
            context_parts.append(f"{source_info}\n{text}")

        return "\n\n".join(context_parts)

    def query(self, question: str) -> RAGResponse:
        """
        Run the full RAG pipeline on a question.

        Steps:
          1. Hybrid retrieve (BM25 + semantic with RRF fusion)
          2. Rerank with cross-encoder
          3. Determine confidence tier from top reranked score
          4. Tier "none"  → return fallback (skip LLM)
             Tier "low"   → call LLM, prepend low-confidence warning
             Tier "full"  → call LLM, return full answer

        Args:
            question: The user's natural-language question.

        Returns:
            A RAGResponse with the answer, sources, and scores.
        """
        print(f"🔍 Query: \"{question}\"")

        # Step 1: Hybrid retrieval (BM25 + semantic + RRF)
        hybrid_results = self.retriever.retrieve_hybrid(question, top_k=self.top_k)

        # Step 2: Rerank with cross-encoder
        reranked = self.reranker.rerank(question, hybrid_results, top_n=self.rerank_top_n)

        # Step 3: Determine confidence tier from top reranked score
        max_score = reranked[0][0] if reranked else 0.0
        tier = get_confidence_tier(max_score)
        print(f"   🎯 Confidence tier: {tier.upper()} (max score: {max_score:.4f})")

        # Extract scores and build source documents for the response
        reranked_scores = [score for score, _text, _meta in reranked]
        source_docs = [
            Document(page_content=text, metadata=meta)
            for _score, text, meta in reranked
        ]

        # Step 4: Generate answer based on tier
        if tier == "none":
            # Tier 1 — No answer: skip LLM entirely
            answer = "I don't have enough information in my documents to answer this question."
            print(f"   ⚠️  No chunks passed relevance threshold — returning fallback answer.")
            has_relevant = False

        elif tier == "low":
            # Tier 2 — Low confidence: call LLM but prepend warning
            context = self._format_context(reranked)
            print(f"   ⚠️  Low confidence — generating with weakly matched context...")
            print(f"   🤖 Generating answer with Llama 3.2...")

            raw_answer = self.chain.invoke({
                "context": context,
                "question": question,
            })
            answer = (
                "[Low confidence — answer based on weakly matched context]\n"
                + raw_answer.strip()
            )
            has_relevant = True

        else:
            # Tier 3 — Full answer: normal flow
            context = self._format_context(reranked)
            print(f"   🤖 Generating answer with Llama 3.2...")

            answer = self.chain.invoke({
                "context": context,
                "question": question,
            })
            has_relevant = True

        return RAGResponse(
            question=question,
            answer=answer.strip(),
            source_documents=source_docs,
            relevance_scores=reranked_scores,
            has_relevant_context=has_relevant,
        )


# ============================================================
# Pretty-print helpers
# ============================================================

def print_response(response: RAGResponse):
    """Pretty-print a RAG response."""
    print(f"\n{'='*70}")
    print(f"❓ Question: {response.question}")
    print(f"{'='*70}")
    print(f"\n💬 Answer:\n{response.answer}")

    if response.has_relevant_context:
        print(f"\n📚 Sources ({len(response.source_documents)} chunks used):")
        for i, (doc, score) in enumerate(
            zip(response.source_documents, response.relevance_scores), 1
        ):
            meta = doc.metadata
            print(
                f"   [{i}] {meta.get('file_name')} | Page {meta.get('page')} | "
                f"Category: {meta.get('category')} | Score: {score:.4f}"
            )
    else:
        print(f"\n⚠️  No relevant sources found in the knowledge base.")

    print(f"{'='*70}\n")


# ============================================================
# Run standalone for testing
# ============================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline()

    # --- Test with in-domain queries ---
    test_queries = [
        # Should work well (in-domain)
        "What is MQTT protocol and what are its key characteristics?",
        "Explain the architecture of IoT networks according to NIST",
        "What are the main features of Arduino UNO R3?",

        # Should trigger "no information" (out-of-domain)
        "What is the capital of India?",
        "Who is Dharshan Kumar?",
    ]

    for question in test_queries:
        response = pipeline.query(question)
        print_response(response)
        print(f"{'#'*70}\n")
