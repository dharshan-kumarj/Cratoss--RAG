# 🚀 RAG-Cratoss

> **Retrieval-Augmented Generation pipeline for IoT & Network Security Documents**

A complete RAG system that ingests IoT/network security PDFs, chunks and embeds them using Sentence Transformers, stores vectors in ChromaDB, and retrieves semantically relevant context for any user query.

---

## 📂 Project Structure

```
RAG-Cratoss/
├── data/
│   └── pdfs/
│       ├── architecture/       # IoT architecture docs (e.g., NIST SP 800-183)
│       ├── hardware/           # Hardware specification docs
│       └── protocols/          # Protocol specs (MQTT, CoAP, etc.)
├── ingestion/
│   ├── loader.py               # Phase 1 — PDF loading with metadata
│   ├── chunker.py              # Phase 2 — Recursive text chunking
│   └── embedder.py             # Phase 3 — Embedding & ChromaDB storage
├── rag/
│   ├── retriever.py            # Phase 4 — Semantic similarity retrieval
│   └── pipeline.py             # Phase 5 — LLM generation (coming soon)
├── vectorstore/
│   └── chroma_db/              # Persisted ChromaDB vector store
├── api/
│   └── main.py                 # FastAPI endpoint (coming soon)
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3 |
| **Framework** | LangChain |
| **PDF Parsing** | PyPDF |
| **Text Splitting** | RecursiveCharacterTextSplitter |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Vector Store** | ChromaDB (persistent) |
| **API** | FastAPI + Uvicorn (planned) |

---

## 🔧 Setup

```bash
# Clone the repo
git clone https://github.com/your-username/RAG-Cratoss.git
cd RAG-Cratoss

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📋 Pipeline Phases & Outputs

### Phase 1 — PDF Loading (`ingestion/loader.py`)

Recursively loads all PDF files from `data/pdfs/` (including subdirectories) and returns LangChain `Document` objects with enriched metadata (source path, category, file name).

```bash
python -m ingestion.loader
```

**What it does:**
- Walks through `architecture/`, `hardware/`, and `protocols/` subdirectories
- Loads each PDF page-by-page using `PyPDFLoader`
- Tags each page with a `category` derived from the subfolder name
- Adds `file_name` to metadata for traceability

---

### Phase 2 — Text Chunking (`ingestion/chunker.py`)

Splits the loaded pages into smaller, overlapping chunks that are better suited for embedding and retrieval.

```bash
python -m ingestion.chunker
```

**Configuration:**
| Parameter | Value |
|-----------|-------|
| `chunk_size` | 1000 characters |
| `chunk_overlap` | 200 characters |
| `separators` | `\n\n` → `\n` → `. ` → ` ` → `""` |

**What it does:**
- Uses `RecursiveCharacterTextSplitter` to keep paragraphs and sentences intact
- Preserves all original metadata (source, page, category, file_name)
- Adds a `chunk_index` to each chunk for traceability

---

### Phase 3 — Embedding & Vector Storage (`ingestion/embedder.py`)

Generates embeddings using Sentence Transformers and stores them in a persistent ChromaDB vector store.

```bash
python -m ingestion.embedder
```

**What it does:**
- Loads the `sentence-transformers/all-MiniLM-L6-v2` model (384-dimensional, normalized embeddings)
- Processes chunks in batches of 50 to avoid memory issues
- Persists the vector store to `vectorstore/chroma_db/`
- Collection name: `rag_cratoss_docs`

**Result:** `577 document chunks` embedded and stored in ChromaDB.

---

### Phase 4 — Retrieval (`rag/retriever.py`) ✅

Loads the persisted ChromaDB vector store and retrieves the most semantically relevant document chunks for any user query.

```bash
python -m rag.retriever
```

**Retrieval Modes:**
| Mode | Method | Description |
|------|--------|-------------|
| Basic | `retrieve(query)` | Top-K similarity search |
| Scored | `retrieve_with_scores(query)` | Returns relevance scores alongside results |
| Filtered | `retrieve_with_filter(query, filter)` | Metadata-filtered search (e.g., by category) |
| LangChain | `get_langchain_retriever()` | Returns a retriever for LangChain pipelines |

#### 🧪 Test Output

**Query 1: `"What is MQTT protocol?"`**

```
======================================================================
🔍 Query: "What is MQTT protocol?"
   Found 5 relevant chunks
======================================================================

📄 Result 1
   File:     mqtt_protocol_spec.pdf
   Category: protocols
   Page:     0
   Chunk #:  437
   ──────────────────────────────────────────────────
   MQTT V3.1 Protocol Specification
   Authors:
   International Business Machines Corporation (IBM)
   Eurotech
   Abstract
   MQ Telemetry Transport (MQTT) is a lightweight broker-based publish/subscribe
   messaging protocol designed to be open, simple, lightweight and easy to implement.
   These characteristics make it...

📄 Result 2
   File:     mqtt_protocol_spec.pdf
   Category: protocols
   Page:     2
   Chunk #:  442
   ──────────────────────────────────────────────────
   1. Introduction
   This specification is split into three main sections:
   the message format that is common to all packet types,
   the specific details of each packet type,
   how the packets flow between client and server.
   Information on how topic wildcards are used is provided in the appendix.

📄 Result 3
   File:     mqtt_protocol_spec.pdf
   Category: protocols
   Page:     6
   Chunk #:  453
   ──────────────────────────────────────────────────
   Protocol name
   The protocol name is present in the variable header of a MQTT CONNECT message.
   This field is a UTF-enc...

📄 Result 4
   File:     mqtt_protocol_spec.pdf
   Category: protocols
   Page:     2
   Chunk #:  443

📄 Result 5
   File:     mqtt_protocol_spec.pdf
   Category: protocols
   Page:     10
   Chunk #:  464
```

**With Relevance Scores:**

| Result | File | Score |
|--------|------|-------|
| #1 | `mqtt_protocol_spec.pdf` (Page 0) | 🎯 **0.6767** |
| #2 | `mqtt_protocol_spec.pdf` (Page 2) | 🎯 **0.6174** |
| #3 | `mqtt_protocol_spec.pdf` (Page 6) | 🎯 **0.5470** |

> ✅ **Observation:** All 5 results correctly come from `mqtt_protocol_spec.pdf` under the `protocols` category. The highest-scoring chunk contains the actual MQTT abstract/definition. Retrieval is **highly accurate** for this query.

---

**Query 2: `"IoT device architecture"`**

```
======================================================================
🔍 Query: "IoT device architecture"
   Found 5 relevant chunks
======================================================================

📄 Result 1
   File:     nist_iot_architecture.pdf
   Category: architecture
   Page:     5
   Chunk #:  13
   ──────────────────────────────────────────────────
   NIST SP 800-183  NETWORKS OF 'THINGS'
   1 Introduction
   From agriculture, to manufacturing, to smart homes, to healthcare, and beyond,
   there is value in having numerous sensory devices connected to la...

📄 Result 2
   File:     nist_iot_architecture.pdf
   Category: architecture
   Page:     3
   Chunk #:  7
   ──────────────────────────────────────────────────
   Abstract
   System primitives allow formalisms, reasoning, simulations, and reliability
   and security risk-tradeoffs to be formulated and argued...

📄 Result 3 — Page 26 | 📄 Result 4 — Page 26 | 📄 Result 5 — Page 5
```

**With Relevance Scores:**

| Result | File | Score |
|--------|------|-------|
| #1 | `nist_iot_architecture.pdf` (Page 5) | 🎯 **0.3987** |
| #2 | `nist_iot_architecture.pdf` (Page 3) | 🎯 **0.3670** |
| #3 | `nist_iot_architecture.pdf` (Page 26) | 🎯 **0.3573** |

> ✅ **Observation:** All 5 results correctly come from `nist_iot_architecture.pdf` under the `architecture` category. The retriever correctly identified the NIST IoT document as the most relevant source.

---

**Query 3: `"network intrusion detection"`**

```
======================================================================
🔍 Query: "network intrusion detection"
   Found 5 relevant chunks
======================================================================

📄 Result 1
   File:     nist_iot_architecture.pdf
   Category: architecture
   Page:     1
   Chunk #:  1

📄 Result 2
   File:     nist_iot_architecture.pdf
   Category: architecture
   Page:     29
   Chunk #:  85

📄 Result 3
   File:     coap_protocol_rfc7252.pdf
   Category: protocols
   Page:     81
   Chunk #:  355

📄 Result 4
   File:     coap_protocol_rfc7252.pdf
   Category: protocols
   Page:     82
   Chunk #:  361

📄 Result 5
   File:     coap_protocol_rfc7252.pdf
   Category: protocols
   Page:     82
   Chunk #:  360
```

**With Relevance Scores:**

| Result | File | Score |
|--------|------|-------|
| #1 | `nist_iot_architecture.pdf` (Page 1) | 🎯 **0.1082** |
| #2 | `nist_iot_architecture.pdf` (Page 29) | 🎯 **0.0209** |
| #3 | `coap_protocol_rfc7252.pdf` (Page 81) | 🎯 **0.0150** |

> ⚠️ **Observation:** Scores are significantly lower (max 0.10) compared to the MQTT query (max 0.67). This suggests the ingested PDFs don't contain dedicated content on "network intrusion detection" — the retriever is pulling the closest matches it can find from security-related sections of the architecture and protocol docs.

---

#### 🧪 Phase 4b — Out-of-Domain Testing

To validate that the retriever can distinguish between relevant and irrelevant queries, we tested with questions **completely unrelated** to the ingested IoT/networking PDFs.

**Query: `"What is Capital of India?"`**

| Result | File | Score |
|--------|------|-------|
| #1 | `coap_protocol_rfc7252.pdf` (Page 3) | ⚠️ **+0.0170** |
| #2 | `coap_protocol_rfc7252.pdf` (Page 72) | ❌ **-0.0059** |
| #3 | `coap_protocol_rfc7252.pdf` (Page 58) | ❌ **-0.0134** |

> ❌ Scores are near-zero/negative. The retriever returned random table-of-contents and URI sections — completely irrelevant noise.

**Query: `"who is Dharshan Kumar"`**

| Result | File | Score |
|--------|------|-------|
| #1 | `coap_protocol_rfc7252.pdf` (Page 21) | ❌ **-0.1302** |
| #2 | `arduino_uno_datasheet.pdf` (Page 10) | ❌ **-0.1925** |
| #3 | `arduino_uno_datasheet.pdf` (Page 0) | ❌ **-0.1990** |

> ❌ All scores are **deeply negative**. The retriever pulled GPIO pin tables and Arduino descriptions — complete garbage, as expected.

**Query: `"How much is LOQ Laptop?"`**

| Result | File | Score |
|--------|------|-------|
| #1 | `coap_protocol_rfc7252.pdf` (Page 17) | ❌ **-0.0398** |
| #2 | `coap_protocol_rfc7252.pdf` (Page 18) | ❌ **-0.0477** |
| #3 | `coap_protocol_rfc7252.pdf` (Page 90) | ❌ **-0.0599** |

> ❌ All scores negative. CoAP option number tables returned — no relevance whatsoever.

---

#### 📊 Score Interpretation Guide

| Score Range | Meaning | Action |
|-------------|---------|--------|
| **> 0.5** | 🟢 High confidence — strong semantic match | Use in LLM context |
| **0.3 – 0.5** | 🟡 Moderate — likely relevant | Use in LLM context |
| **0.1 – 0.3** | 🟠 Weak — tangentially related | Use with caution |
| **< 0.1** | 🔴 No match — out-of-domain query | Reject / "I don't know" |

> 💡 **Key Takeaway:** The retriever **always returns results** (it's just nearest-neighbor search), but the **relevance scores** clearly separate good matches from noise. In the LLM pipeline (Phase 5), we'll use a **score threshold (~0.25)** to filter out irrelevant chunks, allowing the system to say *"I don't have information about this topic"* instead of hallucinating from garbage context.

---

### Phase 5 — LLM Generation Pipeline (`rag/pipeline.py`) 🔜

> *Coming soon* — Will take retrieved chunks and feed them as context to an LLM to generate grounded answers.

### Phase 6 — API Endpoint (`api/main.py`) 🔜

> *Coming soon* — FastAPI REST endpoint to expose the full RAG pipeline.

---

## 📊 Pipeline Summary

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  📄 PDFs     │────▶│  ✂️ Chunker  │────▶│  🤖 Embedder │────▶│  💾 ChromaDB │
│  (loader.py) │     │ (chunker.py) │     │(embedder.py) │     │  577 chunks  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                     ┌──────────────┐     ┌──────────────┐            │
                     │  💬 Answer   │◀────│  🔍 Retrieve │◀───────────┘
                     │(pipeline.py) │     │(retriever.py)│
                     └──────────────┘     └──────────────┘
                            │
                     ┌──────────────┐
                     │  🌐 API      │
                     │  (main.py)   │
                     └──────────────┘
```

---

## 📝 License

This project is for educational and research purposes.
