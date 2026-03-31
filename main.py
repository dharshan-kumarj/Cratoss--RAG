"""
main.py - RAG-Cratoss Interactive Entry Point

Run this from the project root to start an interactive Q&A session
with the RAG pipeline. Type your questions and get answers grounded
in the IoT knowledge base.

Usage:
    python main.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.pipeline import RAGPipeline, print_response


def main():
    """Launch the RAG pipeline and enter an interactive question loop."""

    print("=" * 60)
    print("  🚀  RAG-Cratoss — IoT Knowledge Assistant")
    print("=" * 60)
    print()

    # Initialize the pipeline (loads retriever, reranker, LLM)
    pipeline = RAGPipeline()

    print("-" * 60)
    print("Type your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 60)
    print()

    while True:
        try:
            question = input("❓ Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        # Run the pipeline and display the result
        response = pipeline.query(question)
        print_response(response)


if __name__ == "__main__":
    main()
