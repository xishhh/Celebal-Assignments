# RAG QA System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.36+-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/ChromaDB-0.5+-yellow?logo=chromadb" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Groq-LLM-green?logo=groq" alt="Groq">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

A modular **Retrieval-Augmented Generation (RAG)** system that enables natural-language question answering over your own documents. Upload PDFs, text files, or markdown documents, and ask questions — the system retrieves relevant content and generates grounded, source-cited answers using a large language model.

---

## Features

- **Multi-format document ingest** — PDF, TXT, and Markdown support with automatic text extraction
- **Local embedding pipeline** — Sentence-Transformers (`all-MiniLM-L6-v2`) for privacy-preserving, cost-free vector embeddings
- **LLM-agnostic architecture** — Pluggable backends (Groq default, OpenAI fallback) with zero code changes to switch
- **Persistent vector store** — ChromaDB indexes survive restarts; reset and re-ingest on demand
- **Source-grounded answers** — Every response cites the document and page number it was derived from
- **Conversation history** — Full QA history with expandable retrieved chunks and relevance scores
- **Edge-case hardened** — Handles corrupt PDFs, empty files, missing keys, and network failures gracefully

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Streamlit  │     │ RAGPipeline  │     │    Loader        │
│    (UI)     │────▶│ (Orchestr.)  │────▶│  + Chunker       │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Groq /    │◀────│   generate_  │◀────│   VectorStore    │
│   OpenAI    │     │   answer()   │     │   (ChromaDB)     │
└─────────────┘     └──────────────┘     └──────────────────┘
```

1. **Ingest** — Documents are loaded, recursively chunked by paragraph/sentence/word boundaries, and indexed into ChromaDB with local embeddings.
2. **Query** — A user question is embedded and used to retrieve the top-K most similar chunks via cosine similarity.
3. **Generate** — Retrieved chunks are formatted into a prompt and sent to the LLM, which answers exclusively from the provided context.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web framework | [Streamlit](https://streamlit.io) |
| Vector database | [ChromaDB](https://www.trychroma.com) (persistent, HNSW) |
| Embedding model | [Sentence-Transformers](https://sbert.net) (`all-MiniLM-L6-v2`) |
| LLM (default) | [Groq](https://groq.com) (`llama-3.1-8b-instant`) |
| LLM (fallback) | [OpenAI](https://openai.com) (`gpt-4o-mini`) |
| PDF parsing | [pypdf](https://pypdf.readthedocs.io) |
| Runtime | Python 3.12+ |

---

## Prerequisites

- Python 3.12 or higher
- An API key for at least one LLM provider:
  - [Groq](https://console.groq.com/keys) (free tier available) — recommended
  - [OpenAI](https://platform.openai.com/api-keys) — optional fallback

---

## Quick Start

### 1. Clone and enter the project

```bash
git clone https://github.com/your-username/rag-qa-system.git
cd rag-qa-system
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env          # macOS / Linux
copy .env.example .env        # Windows
```

Edit `.env` and add your API key(s):

```ini
GROQ_API_KEY="gsk_your_key_here"
OPENAI_API_KEY="sk-your-key-here"    # only if using OpenAI
```

### 5. Run the application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

All tunable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 512 | Maximum characters per chunk |
| `CHUNK_OVERLAP` | 50 | Character overlap between consecutive chunks |
| `TOP_K` | 3 | Number of chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformer model for embeddings |
| `LLM_PROVIDER` | `groq` | Active LLM backend (`groq` or `openai`) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model identifier |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model identifier |
| `COLLECTION_NAME` | `rag_docs` | ChromaDB collection name |

---

## Usage

1. **Upload documents** — Use the sidebar file uploader (PDF, TXT, MD supported).
2. **Ingest** — Click *Ingest documents*. Optionally check *Reset vector store* to re-index from scratch.
3. **Ask questions** — Type a question in the main input and click *Ask*.
4. **Review answers** — Each answer cites its sources. Expand *Retrieved chunks* to inspect the raw context and relevance scores.

---

## Project Structure

```
rag_qa_system/
├── app.py              # Streamlit UI — upload, ingest, QA interface
├── config.py           # Central configuration and environment loading
├── loader.py           # Document loading (PDF, TXT, MD)
├── chunker.py          # Recursive text chunking (paragraph → sentence → word)
├── vectorstore.py      # ChromaDB wrapper — indexing and semantic search
├── llm.py              # LLM abstraction — Groq and OpenAI backends
├── rag_pipeline.py     # Pipeline orchestrator — ingest and query workflows
├── data/               # Sample documents for testing
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
└── README.md           # This file
```

---

## How It Works

### Document Loading (`loader.py`)
Files are dispatched by extension: PDFs are parsed page-by-page with `pypdf`; TXT and MD files are read as plain text. Corrupt or unreadable files are skipped with a warning — the pipeline continues uninterrupted.

### Chunking (`chunker.py`)
A custom recursive algorithm splits documents at natural boundaries:
- **First pass** — split by paragraph (`\n\n`)
- **Second pass** — split by sentence (`.!?` followed by whitespace)
- **Final pass** — split by word boundary (words are never broken)

Chunks are then merged with configurable overlap to preserve context across boundaries.

### Vector Indexing (`vectorstore.py`)
Each chunk is embedded locally using Sentence-Transformers and stored in a persistent ChromaDB collection. Batches of 100 chunks are indexed at a time for performance.

### Retrieval & Generation (`llm.py`, `rag_pipeline.py`)
On query, the top-K most similar chunks are retrieved via cosine similarity. They are formatted into a prompt with source labels and sent to the LLM. The system prompt enforces grounded answering — the model must answer exclusively from the provided context and cite its sources, or decline if information is insufficient.

---

## Roadmap

- [ ] Cohere reranking integration for improved retrieval precision
- [ ] Streaming LLM responses for real-time UX
- [ ] Multi-document conversation with cross-document synthesis
- [ ] Docker deployment with docker-compose
- [ ] Export chat history to JSON / Markdown


