# 🚀 RAG-Cratoss

> **Retrieval-Augmented Generation Pipeline for IoT & Network Security Documents**

A complete, end-to-end RAG system that ingests IoT/network security PDFs, chunks and embeds them using Sentence Transformers, stores vectors in ChromaDB, retrieves semantically relevant context, and generates grounded answers using **Meta's Llama 3.2** running locally via **Ollama** — all without any API keys or cloud dependencies.

---

## 🎯 What Is This Project?

**RAG-Cratoss** is a domain-specific question-answering system built on the Retrieval-Augmented Generation (RAG) architecture. Instead of relying on a general-purpose LLM that may hallucinate, it:

1. **Ingests** your own PDF documents (IoT specs, protocol RFCs, hardware datasheets)
2. **Splits** them into semantically meaningful chunks
3. **Embeds** each chunk into a 384-dimensional vector space
4. **Stores** them in a persistent vector database (ChromaDB)
5. **Retrieves** the most relevant chunks when you ask a question
6. **Generates** an accurate, grounded answer using only the retrieved context

> 💡 **Key Insight:** The LLM never makes up answers — it only responds based on your documents. If no relevant information is found, it says *"I don't have enough information."*

---

## 🧠 How It Works

```
                         📄 Your PDFs (IoT, Protocols, Hardware)
                                        │
                         ┌──────────────▼──────────────┐
                         │   Phase 1: PDF Loading       │
                         │   (loader.py — PyPDF)        │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   Phase 2: Text Chunking     │
                         │   (chunker.py — 1000 chars)  │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   Phase 3: Embedding         │
                         │   (embedder.py — MiniLM)     │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   Phase 4: Vector Storage    │
                         │   (ChromaDB — 577 chunks)    │
                         └──────────────┬──────────────┘
                                        │
         User Question ─────────────────┤
                                        │
                         ┌──────────────▼──────────────┐
                         │   Phase 5: Retrieval         │
                         │   (retriever.py — Top-K)     │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   Phase 6: Score Filtering   │
                         │   (threshold ≥ 0.25)         │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   Phase 7: LLM Generation    │
                         │   (Llama 3.2 via Ollama)     │
                         └──────────────┬──────────────┘
                                        │
                                   💬 Answer
                              (with source citations)
```

---

## 📂 Project Structure

```
RAG-Cratoss/
├── data/
│   └── pdfs/
│       ├── architecture/          # IoT architecture docs (NIST SP 800-183)
│       ├── hardware/              # Hardware datasheets (Arduino UNO R3)
│       └── protocols/             # Protocol specs (MQTT, CoAP RFC 7252)
├── ingestion/
│   ├── __init__.py
│   ├── loader.py                  # Phase 1 — PDF loading with metadata
│   ├── chunker.py                 # Phase 2 — Recursive text chunking
│   └── embedder.py                # Phase 3 — Embedding & ChromaDB storage
├── rag/
│   ├── __init__.py
│   ├── retriever.py               # Phase 4 — Semantic similarity retrieval
│   └── pipeline.py                # Phase 5 — Full RAG pipeline (retrieve + generate)
├── vectorstore/
│   └── chroma_db/                 # Persisted ChromaDB vector store (577 chunks)
├── api/
│   └── main.py                    # FastAPI endpoint (planned)
├── .env                           # Environment variables (optional)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Technology | Details |
|-----------|------------|---------|
| **Language** | Python 3.11+ | |
| **Framework** | LangChain | Orchestration & chain building |
| **PDF Parsing** | PyPDF | Page-by-page PDF extraction |
| **Text Splitting** | RecursiveCharacterTextSplitter | 1000-char chunks with 200-char overlap |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional, normalized embeddings |
| **Vector Store** | ChromaDB (persistent) | Local storage, cosine similarity |
| **LLM** | Meta Llama 3.2 (3B) | Running locally via Ollama |
| **LLM Runtime** | Ollama | Local inference, GPU accelerated |
| **API** | FastAPI + Uvicorn | REST endpoint (planned) |

---

## 🔧 Setup & Installation

### Prerequisites

- **Python 3.11+**
- **Ollama** (for running Llama 3.2 locally)
- **NVIDIA GPU** (optional but recommended — e.g., RTX 4060)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/RAG-Cratoss.git
cd RAG-Cratoss
```

### Step 2: Create Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Install Ollama & Download Llama 3.2

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download Llama 3.2 (3B model, ~2GB)
ollama pull llama3.2
```

### Step 4 (Optional): GPU Setup for WSL2

If you have an NVIDIA GPU and are using WSL2:

```bash
# Verify GPU is visible in WSL
nvidia-smi

# Restart Ollama to detect GPU
sudo systemctl restart ollama

# Verify GPU is being used
ollama ps
```

> 💡 **Performance:** With an RTX 4060, responses take **5-10 seconds**. On CPU, expect **1-3 minutes** per query.

---

## 🚀 Usage

### Run the Full Pipeline

```bash
# Make sure Ollama is running
ollama serve &

# Run the RAG pipeline
python rag/pipeline.py
```

This will:
1. Load the ChromaDB vector store (577 embedded document chunks)
2. Initialize Llama 3.2 locally via Ollama
3. Run 5 test queries (3 in-domain + 2 out-of-domain)
4. Print answers with source citations

### Run Individual Phases

```bash
# Phase 1: Load PDFs
python -m ingestion.loader

# Phase 2: Chunk documents
python -m ingestion.chunker

# Phase 3: Embed & store in ChromaDB
python -m ingestion.embedder

# Phase 4: Test retrieval only
python -m rag.retriever

# Phase 5: Full RAG pipeline (retrieve + generate)
python rag/pipeline.py
```

---

## 📋 Pipeline Phases in Detail

### Phase 1 — PDF Loading (`ingestion/loader.py`)

Recursively loads all PDF files from `data/pdfs/` and returns LangChain `Document` objects with enriched metadata.

- Walks through `architecture/`, `hardware/`, and `protocols/` subdirectories
- Loads each PDF page-by-page using `PyPDFLoader`
- Tags each page with a `category` derived from the subfolder name
- Adds `file_name` to metadata for traceability

---

### Phase 2 — Text Chunking (`ingestion/chunker.py`)

Splits loaded pages into smaller, overlapping chunks optimized for embedding and retrieval.

| Parameter | Value |
|-----------|-------|
| `chunk_size` | 1000 characters |
| `chunk_overlap` | 200 characters |
| `separators` | `\n\n` → `\n` → `. ` → ` ` → `""` |

- Uses `RecursiveCharacterTextSplitter` to keep paragraphs and sentences intact
- Preserves all original metadata (source, page, category, file_name)
- Adds a `chunk_index` to each chunk for traceability

---

### Phase 3 — Embedding & Storage (`ingestion/embedder.py`)

Generates embeddings using Sentence Transformers and persists them in ChromaDB.

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional)
- Processes chunks in batches of 50 to avoid memory issues
- Persists the vector store to `vectorstore/chroma_db/`
- Collection name: `rag_cratoss_docs`
- **Result:** 577 document chunks embedded and stored

---

### Phase 4 — Retrieval (`rag/retriever.py`)

Loads the persisted ChromaDB vector store and retrieves the most semantically relevant chunks for any user query.

**Retrieval Modes:**

| Mode | Method | Description |
|------|--------|-------------|
| Basic | `retrieve(query)` | Top-K similarity search |
| Scored | `retrieve_with_scores(query)` | Returns relevance scores alongside results |
| Filtered | `retrieve_with_filter(query, filter)` | Metadata-filtered search (e.g., by category) |
| LangChain | `get_langchain_retriever()` | Returns a retriever for LangChain pipelines |

---

### Phase 5 — Generation Pipeline (`rag/pipeline.py`)

The core orchestration layer that connects retrieval to generation with relevance filtering.

**Pipeline Flow:**

```
User Question
     │
     ▼
┌─────────────────┐
│ Retrieve Top-5  │  ← ChromaDB semantic search
│ chunks          │
└────────┬────────┘
         │
     ▼
┌─────────────────┐
│ Filter by score │  ← Discard chunks with score < 0.25
│ threshold       │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Has context?  No context?
    │         │
    ▼         ▼
┌─────────┐  ┌──────────────────────┐
│ Format  │  │ Return: "I don't     │
│ context │  │ have enough info..." │
│ + LLM   │  └──────────────────────┘
│ generate│
└────┬────┘
     │
     ▼
💬 Answer with source citations
```

**Key Configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LLM_MODEL` | `llama3.2` | Meta's Llama 3.2 3B via Ollama |
| `MAX_NEW_TOKENS` | 512 | Maximum tokens in generated answer |
| `TEMPERATURE` | 0.3 | Low temperature for factual answers |
| `RELEVANCE_THRESHOLD` | 0.25 | Chunks scoring below this are discarded |
| `TOP_K` | 5 | Number of chunks retrieved per query |

**Prompt Strategy:** The LLM is instructed to act as an IoT/network security expert and answer **only** from the provided context. If the context is insufficient, it explicitly says so — preventing hallucination.

---

## 📊 Retrieval Quality & Score Guide

| Score Range | Meaning | Pipeline Action |
|-------------|---------|-----------------|
| **> 0.5** | 🟢 High confidence — strong semantic match | ✅ Use in LLM context |
| **0.3 – 0.5** | 🟡 Moderate — likely relevant | ✅ Use in LLM context |
| **0.1 – 0.3** | 🟠 Weak — tangentially related | ⚠️ Use with caution |
| **< 0.1** | 🔴 No match — out-of-domain query | ❌ Reject → "I don't know" |

### Example Results

**In-domain query:** *"What is MQTT protocol?"*
| Result | Source File | Score |
|--------|------------|-------|
| #1 | `mqtt_protocol_spec.pdf` (Page 0) | 🟢 **0.6767** |
| #2 | `mqtt_protocol_spec.pdf` (Page 2) | 🟢 **0.6174** |
| #3 | `mqtt_protocol_spec.pdf` (Page 6) | 🟢 **0.5470** |

> ✅ All results correctly from the MQTT spec. High scores = high confidence.

**Out-of-domain query:** *"What is the capital of India?"*
| Result | Source File | Score |
|--------|------------|-------|
| #1 | `coap_protocol_rfc7252.pdf` (Page 3) | 🔴 **+0.0170** |
| #2 | `coap_protocol_rfc7252.pdf` (Page 72) | 🔴 **-0.0059** |

> ❌ Near-zero/negative scores. Pipeline correctly rejects these and returns *"I don't have enough information."*

---

## 📚 Knowledge Base Documents

The system currently has **577 embedded chunks** from these documents:

| Category | Document | Description |
|----------|----------|-------------|
| **Architecture** | `nist_iot_architecture.pdf` | NIST SP 800-183 — Networks of 'Things' |
| **Hardware** | `arduino_uno_datasheet.pdf` | Arduino UNO R3 full datasheet |
| **Protocols** | `mqtt_protocol_spec.pdf` | MQTT V3.1 Protocol Specification |
| **Protocols** | `coap_protocol_rfc7252.pdf` | CoAP — RFC 7252 (Constrained Application Protocol) |

> 💡 **Add your own PDFs:** Simply drop PDF files into `data/pdfs/<category>/` and re-run the ingestion pipeline (Phases 1-3).

---

## 🛠️ Configuration

All pipeline parameters can be tuned in the respective files:

### `rag/pipeline.py`
```python
LLM_MODEL = "llama3.2"          # Change to "llama3.2:1b" for faster CPU inference
MAX_NEW_TOKENS = 512            # Max response length
TEMPERATURE = 0.3               # Higher = more creative, Lower = more factual
RELEVANCE_THRESHOLD = 0.25      # Stricter = fewer but higher quality chunks
TOP_K = 5                       # Number of chunks to retrieve
```

### `ingestion/chunker.py`
```python
CHUNK_SIZE = 1000               # Characters per chunk
CHUNK_OVERLAP = 200             # Overlap between consecutive chunks
```

---

## 🗺️ Roadmap

- [x] **Phase 1** — PDF Loading with metadata enrichment
- [x] **Phase 2** — Recursive text chunking
- [x] **Phase 3** — Sentence Transformer embeddings + ChromaDB storage
- [x] **Phase 4** — Semantic retrieval with relevance scoring
- [x] **Phase 5** — LLM generation pipeline (Llama 3.2 via Ollama)
- [ ] **Phase 6** — FastAPI REST endpoint (`api/main.py`)
- [ ] **Phase 7** — Web UI / Chat interface
- [ ] **Phase 8** — Evaluation metrics (BLEU, ROUGE, faithfulness)
- [ ] **Phase 9** — Multi-document conversation memory

---

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **Storage** | 5 GB | 10 GB |
| **GPU** | None (CPU works) | NVIDIA RTX 4060+ (8GB VRAM) |
| **OS** | Ubuntu 20.04 / WSL2 | Ubuntu 22.04 / WSL2 |
| **Python** | 3.10 | 3.11+ |

---

## 📝 License

This project is for educational and research purposes.

---

## 👤 Author

**Dharshan Kumar J**

Built as a hands-on project to learn and implement Retrieval-Augmented Generation from scratch.
