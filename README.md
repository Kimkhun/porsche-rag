# Porsche RAG — Knowledge Base

A Retrieval-Augmented Generation system for querying a Porsche knowledge base. Ask questions about models, history, technology, and motorsport — the system retrieves relevant documents and generates grounded answers using Gemini.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires a GCP service account key at `gcp-key.json` in the project root (for Vertex AI embeddings + Gemini).

## Features

| Feature | Description |
|---|---|
| **Hybrid search** | BM25 keyword + vector embedding search fused via Reciprocal Rank Fusion |
| **Semantic chunking** | Paragraph-aware document splitting (not blind 80-word cuts) |
| **Date-aware answers** | Sources tagged with dates (from Wikipedia API or text); LLM warns if info may be outdated |
| **Cross-encoder reranking** | Re-ranks retrieved chunks by query relevance using a MiniLM model |
| **Conversation history** | Previous turns used for context and query rewrite |
| **Query rewriting** | Gemini rewrites follow-up questions (pronouns disambiguated) before retrieval |
| **Wikipedia dates** | Auto-fetches last-modified timestamps for Wikipedia articles via API |
| **Timing breakdown** | Per-query latency shown under each answer (rewrite / retrieve / generate) |

## Project structure

```
final_project_starter/
├── app.py                  # Streamlit UI (dark theme, chat interface)
├── requirements.txt
├── gcp-key.json            # GCP service account (not committed)
├── data/
│   ├── sample_docs/        # 121 Wikipedia articles about Porsche
│   └── index/              # ChromaDB vectors + Wikipedia date cache (gitignored)
└── rag/
    ├── ingest.py           # Load, chunk, extract dates from documents
    ├── embed_store.py      # VectorStore (ChromaDB + BM25 + reranker)
    └── generate.py         # Prompt building + Gemini streaming
```

## How it works

### Search pipeline

```
User query → rewrite_search_query (Gemini)
           → VectorStore.query (hybrid BM25 + vector)
           → Cross-encoder reranker
           → llm_answer_stream (Gemini)
           → Display answer + sources + timings
```

### Date extraction

Dates are extracted in priority order:
1. **Filename pattern**: `Porsche_Newsroom_2025-03-15.txt` → `2025-03-15`
2. **Text content**: "As of 2024", "Introduced in 2023", "January 15, 2023", etc.
3. **Wikipedia API**: Falls back to last-modified timestamp of the matching Wikipedia article (batched, cached locally)

### Hybrid search

When enabled, both retrieval methods run in parallel:
- **Vector**: Query embedded via `gemini-embedding-001`, searched in ChromaDB via cosine similarity
- **BM25**: Query token-matched against corpus using Okapi BM25 (k1=1.5, b=0.75)

Results fused via RRF: `score = 1/(60 + rank_vector) + 1/(60 + rank_bm25)`

## Document format

Place `.txt` or `.pdf` files in `data/sample_docs/`. The system loads them on startup.

**For date detection**, name files with ISO dates or include dates in the first 500 characters:
- `Porsche_Macan_EV_2025-03-15.txt`
- Content starting with "As of 2024, the Porsche Macan..."

Wikipedia articles work out of the box — their last-modified dates are fetched automatically via the Wikipedia API.

## Sidebar controls

| Control | Default | Purpose |
|---|---|---|
| Chunks retrieved | 3 | Number of source chunks to fetch |
| Rerank results | On | Cross-encoder reranking of results |
| Hybrid search | On | BM25 + vector fusion (vs pure vector) |

## Timing

Each answer shows a breakdown:
```
rewrite: 1.2s | retrieve: 0.8s | generate: 2.1s | total: 4.1s
```

- **rewrite**: Gemini query rewriting (can be skipped for simple queries)
- **retrieve**: Embedding + ChromaDB search + BM25 + reranking
- **generate**: Gemini answer streaming

## Adding more documents

### From Porsche Newsroom

The official press site (newsroom.porsche.com) has hundreds of articles. Save them as `.txt` files with dates in the filename:
```
Porsche_Newsroom_2025-03-15.txt
```

### From Wikipedia

The existing 121 articles came from Wikipedia. Use the Wikipedia API to fetch more:
```
https://en.wikipedia.org/api/rest_v1/page/summary/Porsche_911_GT3
```

## Tech stack

- **UI**: Streamlit
- **Vector DB**: ChromaDB (persistent, cosine similarity)
- **Embeddings**: Vertex AI `gemini-embedding-001`
- **LLM**: Gemini 3.5 Flash via Vertex AI
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Search**: BM25 (custom implementation) + vector hybrid

## Git

Initialized with `.gitignore` excluding `gcp-key.json`, `__pycache__/`, `data/index/`, and `.env`.
