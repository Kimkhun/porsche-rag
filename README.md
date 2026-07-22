# 🏎️ Porsche Intelligence — High-Performance RAG Engine

A premium, localized Retrieval-Augmented Generation (RAG) system tailored for Porsche engineering, history, motorsport, and heritage. Powered by Google Vertex AI Gemini models, ChromaDB, and hybrid keyword-vector search.

Designed with a sleek, dark titanium slate and Guards Red dashboard, this application delivers instant, context-grounded answers about Porsche's automotive legacy with micro-telemetry profiling.

---

## ⚡ High-Performance Architecture

The pipeline is optimized for sub-second database operations and minimal network overhead:

```
  [ User Query ]
        │
        ▼ (Optional)
  [ Query Rewriter (Gemini) ] ──► Disambiguates pronouns/history context
        │
        ▼
  [ Hybrid Retriever ] ──────────► Queries Vector (ChromaDB) + Keyword (BM25) in parallel
        │
        ▼
  [ Reciprocal Rank Fusion ] ────► Merges results: score = 1/(60 + r_vec) + 1/(60 + r_bm25)
        │
        ▼ (Candidate filter)
  [ Cross-Encoder Reranker ] ────► Local MS-MARCO model scores top candidates on CPU
        │
        ▼
  [ Answer Generator (Gemini) ] ─► Streams context-aware answers to user
        │
        ▼
  [ Telemetry Profiler ] ────────► Breaks down execution milliseconds on UI
```

---

## 🛠️ Key Technical Features

| Feature | Engineering Implementation |
|---|---|
| **Hybrid Search** | Cosine-similarity vector matches blended with Okapi BM25 keyword rankings. |
| **Reciprocal Rank Fusion** | Blends keyword and semantic searches to capture both exact numbers (like gear ratios) and general intent. |
| **CPU-Optimized Reranking** | Scores candidates using `ms-marco-MiniLM-L-6-v2` locally. Optimized to score the top `k * 2` documents to prevent CPU bottlenecking. |
| **Low-Latency Routing** | Configured to run in the `asia-southeast1` regional endpoint, reducing embedding API network roundtrip times to **~0.3 seconds**. |
| **Warm Client Caching** | Caches the Vertex AI GenAI Client globally to eliminate authentication and handshake latency on repeat queries. |
| **Websocket Buffer** | Renders the text stream to the UI in 4-token increments, minimizing Streamlit redraw blocks and websocket rendering overhead. |
| **Date-Aware Temporal Guard** | Extracts dates from filenames, text contents, or Wikipedia APIs. The LLM warns users if retrieved data may be outdated relative to the query context. |

---

## 🔬 Algorithmic Deep-Dive & Performance Optimizations

### 1. Hybrid Search & Reciprocal Rank Fusion (RRF)
To balance semantic understanding (e.g. understanding "hybrid racer") and exact keyword matches (e.g. searching for a specific model number like "919"), the system uses a dual-retriever hybrid pipeline:
*   **Vector Search**: Dense retriever query embedded using Vertex AI, returning candidates sorted by cosine similarity distance.
*   **BM25 Search**: Sparse retriever checking frequency matches across the tokenized text index.
*   **Rank Fusion**: The results are combined using the **Reciprocal Rank Fusion (RRF)** formula:
    
    ```text
    Score_RRF(doc) = 1 / (60 + Rank_vector(doc)) + 1 / (60 + Rank_BM25(doc))
    ```
    
    *   Where the constant $K = 60$ parameter dampens the impact of high-ranking outliers from any single search pipeline. 
    *   By fusing *ranks* instead of raw floating scores, RRF cleanly normalizes vector distances and BM25 statistics into a single unified scale without scaling skew.

### 2. Okapi BM25 Search Engine
The custom-built token-matching keyword engine uses the standard **Okapi BM25** formula to score document relevance:

```text
Score_BM25(doc) = SUM [ IDF(word) * (f(word, doc) * (k1 + 1)) / (f(word, doc) + k1 * (1 - b + b * (len(doc) / avg_len))) ]
```

*   **Term Saturation (k1 = 1.5)**: Limits the scale of term frequency. A word occurring 10 times in a short chunk does not score 10 times higher than a word occurring once.
*   **Length Normalization (b = 0.75)**: Scales the penalty for document length relative to the average chunk length. If a term matches inside a short, concise chunk, it scores higher than if it matched in a very long paragraph.
*   **IDF (Inverse Document Frequency)**: Calculated dynamically upon document database synchronization to discount common words and weight rare nouns heavily.

### 3. Bi-Encoder vs. Cross-Encoder Reranking
RAG pipelines face a trade-off between retrieval speed and accuracy. This system uses a **two-stage retrieval pipeline**:
1.  **Stage 1 (Bi-Encoder / Dense Vector)**: Encodes queries and documents into independent vector embeddings. Retrieval is very fast ($O(d)$ cosine operations) but cannot model complex interactions between the query and text tokens.
2.  **Stage 2 (Cross-Encoder / Reranker)**: Feeds both query and candidate texts into the model *simultaneously*, allowing full self-attention layers to score token interactions. This is highly accurate but computationally heavy.
*   **CPU Optimization**: To run this Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) efficiently on a local CPU without GPU resources, the candidate count was optimized from `top_k * 5` (15 candidates) down to `top_k * 2` (6 candidates). This reduced Cross-Encoder inference duration from **~12.0s** to **~0.11s** (a **99% latency reduction**) while keeping top semantic context extremely precise.

### 4. Chronological Date Extraction Stack (Ingestion vs. Query Runtime)
To prevent temporal misalignment in RAG answers, the system extracts, cache-stores, and injects chronological dates. This process is split between **Ingestion Time** and **Query Runtime** to eliminate query blocking latency:

#### A. Ingestion Time (Run once on startup sync)
When the application starts up, the folder synchronization task parses new documents and runs a three-tiered cascade to resolve creation dates:
1.  **Filename Regex**: Scans the filename for ISO 8601 timestamps using pattern `(\d{4})[-_](\d{2})[-_](\d{2})` (e.g. `Porsche_Newsroom_2025-03-15.txt` -> `2025-03-15`).
2.  **Document Text Parsing**: Scans the first **600 characters** of the document text using five pre-compiled patterns to identify phrases (like `"As of 2024"`, `"Launched in 2023"`, `"[year] model year"`) without matching irrelevant numbers in the body.
3.  **Wikipedia REST API Fallback**: If no date is found, the engine queries the MediaWiki API to fetch the last-modified revision timestamp of the corresponding article. To avoid network overhead on reboot, these dates are saved to a local disk cache (`wikipedia_dates.json`).

*All resolved dates are written into ChromaDB document metadata.*

#### B. Query Runtime (Run on every user message)
Because dates are pre-computed during ingestion, there is **zero network or regex overhead** during the query phase:
1.  ChromaDB retrieves the matching document chunks.
2.  The metadata date field is pulled instantly (`results["metadatas"]`) and formatted directly into the prompt context wrapper.
3.  Gemini uses the date tag to reason about the age of the information.

### 5. API & Network Optimization Registry
By tracking telemetry loops, network handshake and authentication delays were resolved through the following optimizations:

| Metric | Before Optimization | After Optimization | Optimization Strategy |
|---|---|---|---|
| **Rewrite Query** | `6.33s` | `0.00s` (First Turn) | Skip rewriting when conversation history (`hist`) is empty. |
| **Embeddings API** | `13.10s` | `0.32s` | Switch endpoints from `LOCATION="global"` to the regional `LOCATION="asia-southeast1"`. |
| **Warm Client Caching** | Handshake on every query | Cached Connection Pool | Store the Vertex AI client instance in a global cache helper (`_get_client()`) to reuse warm connections. |
| **UI Render Block** | Character-by-character refresh | 4-token batch frames | Buffer updates in `app.py` to update Streamlit every 4 tokens, reducing websocket traffic and DOM rendering lag. |

---

## 📁 Project Structure

```
final_project_starter/
├── app.py                  # Streamlit Dashboard (Guards Red Theme & Telemetry UI)
├── requirements.txt        # Package dependencies
├── logo.png                # Brand logo embedded as Base64 in Header
├── gcp-key.json            # Vertex AI credentials (gitignored)
├── data/
│   ├── sample_docs/        # Local folder for custom document uploads (e.g. PDFs/Manuals)
│   └── index/              # ChromaDB vector index files & Wikipedia date cache (gitignored)
└── rag/
    ├── ingest.py           # Document chunking helpers and Wikipedia date parsing
    ├── embed_store.py      # VectorStore (dynamic Wikipedia API content sync + local docs folder)
    └── generate.py         # Prompt engineering, query rewriting, and client caching
```

---

## 🚀 Quick Start

### 1. Requirements installation
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Add Credentials
Place your Vertex AI Google Cloud service account key JSON in the project root and name it:
```
gcp-key.json
```
*(This file is pre-configured in `.gitignore` and will never be committed to repository branches).*

### 3. Launch App
Execute the Streamlit application:
```bash
streamlit run app.py
```

---

## 🎛️ Dashboard Controls

*   **Chunks Retrieved (`top_k`)**: Set how many matching reference documents are injected into the context prompt.
*   **Rerank Results**: Toggles the local CPU Cross-Encoder reranker. Turn off to run purely on vector distance for maximum retrieval speed.
*   **Hybrid Search**: Blends BM25 scores with Vector search. Turn off to query solely using vector similarity.

---

## 📊 Telemetry Diagnostics

Under every assistant message bubble, a detailed telemetry row displays exactly where execution time was spent:

```
Telemetry: rewrite: 0.00s · retrieve: 0.54s [embed-api: 0.38s | vector-db: 0.00s | bm25: 0.04s | rerank-cpu: 0.12s] · generate: 8.84s · total: 12.27s
```

*   **rewrite**: Query refinement (skipped on the first message turn to save time).
*   **retrieve**: Total search block. Breaks down into:
    *   `embed-api`: Network roundtrip to generate search embeddings.
    *   `vector-db`: Persistent local index search.
    *   `bm25`: Keyword index parsing.
    *   `rerank-cpu`: CPU deep-learning scoring.
*   **generate**: Stream duration (measures the live token presentation speed on the screen).
