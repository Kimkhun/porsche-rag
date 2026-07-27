# Evaluation — Porsche RAG Knowledge Base

Run on 2026-07-22 after fresh sync. 111 Wikipedia articles, 5654 chunks, hybrid search (BM25 + vector), cross-encoder reranking, Gemini 3.5 Flash.

## Test Queries

| # | Query | Type | Top sources retrieved | Relevant? | Answer quality |
|---|-------|------|---------------------|-----------|----------------|
| 1 | Tell me about the history of Porsche | Broad overview | Porsche 908, Porsche Family, Porsche In Motorsport | ⚠️ Partial — retrieved a race car article (908) instead of general history | Decent but missed the main "Porsche" article; the Family and Motorsport sources were useful |
| 2 | What is the fastest Porsche ever made? | Comparison | 911 GT3, 911, 919 Hybrid | ✅ Good sources — 919 Hybrid is the fastest, GT3 is the fastest production | Correctly identified 919 Hybrid for track, GT3 RS for production; good nuance |
| 3 | How do Porsche engines differ from other sports car engines? | Technical | Carrera GT, Porsche 907, Porsche Unseen | ❌ None were the right sources. Should have found the engine articles (flat-six, V8) | The answer was generic; the correct engine-specific articles weren't retrieved |
| 4 | What electric vehicles does Porsche currently produce? | Current lineup | Porsche, List of Porsche vehicles, Ruf Automobile | ✅ Actually correct! The generic Porsche article covers the full EV lineup (Mission E, Taycan, Macan EV). Specific model articles only cover one car each | The generic article answered this correctly — Taycan and Macan articles are too specific for a lineup question |
| 5 | Who were the key people behind Porsche's success? | Biography | Porsche, Ferdinand Porsche, Porsche Family | ✅ Perfect | Gave a solid answer covering Ferry, Ferdinand, Piëch, and the family |
| 6 | What motorsport achievements does Porsche have? | Motorsport | 911 RSR, Porsche In Motorsport, Porsche 962 | ✅ Good | Answered with Le Mans wins, 917 dominance, 956/962 era — correct and detailed |
| 7 | Explain the difference between the Cayenne and Macan | Comparison | Porsche Macan, Porsche Cayenne, Porsche Macan (duplicate) | ✅ Correct articles retrieved | Good comparison between the two SUVs, covered size, engine options, target market |
| 8 | What is the most reliable Porsche model? | Subjective | Porsche (×3 — all same article!) | ⚠️ Retrieval returned the same source 3 times instead of diverse results | Answered but from limited perspective — no specific reliability data exists in the corpus |
| 9 | How has the Porsche 911 design evolved? | Evolution | Porsche 911 (996), Porsche In Motorsport, Porsche Unseen | ⚠️ Got the 996 generation article which is good, but missed the main 911 and classic articles | Reasonable answer on design evolution but missed earlier generations |
| 10 | Does Porsche manufacture motorcycles or boats? | Out-of-domain | Porsche, Porsche In Motorsport, Taycan | ✅ Correctly had no direct source | Said "no" confidently — good graceful failure. Noted that Porsche only makes cars |
| 11 | What is the price of a new Porsche 911? | Current pricing | Singer Vehicle Design, 911 (991), Boxster and Cayman | ❌ Singer is a resto-mod shop, not relevant to pricing | Correctly said it doesn't have pricing info — good graceful failure. Flagged sources may be outdated |
| 12 | Tell me about Porsche's involvement in Formula One | Historical motorsport | Porsche 3512, Porsche In Motorsport, Porsche 753 engine | ✅ Good — 3512 is the V12 F1 engine, 753 is the early F1 engine | Detailed answer about the 804 F1 car, the 3512 project, and the TAG-Porsche partnership |

## Write-up

### Retrieval Quality

Hybrid search (BM25 + vector) works well for both specific and broad queries. Specific model queries (Taycan, Macan EV, 911 GT3) return the correct articles at #1 with high reranker scores. Broad list queries ("what electric vehicles") correctly return the generic Porsche article which covers the full lineup — specific model articles are not the right answer for a lineup question. The cross-encoder reranker effectively filters out weak matches (negative scores = irrelevant).

The main retrieval gap is query 3 (engines) — the engine-specific articles exist but weren't ranked highly enough. This may be a chunking issue: the engine articles are short and their first few chunks may not clearly signal they're about engine design philosophy.

Duplicate chunks from the same document sometimes appear multiple times in results (query 8 returned 3× the same `Porsche` article). The reranker helps somewhat but deduplication at the document level before passing to the LLM would improve answer diversity.

### Generation Quality

The LLM gives grounded, conversational answers that cite sources. It handles missing information well — queries 10, 11, and parts of query 4 correctly said "I don't have that information" instead of hallucinating. The "outdated" date warnings (query 11) work correctly. Weaknesses: the conversational tone is repetitive ("Hey there fellow Porsche enthusiast" every time), and when retrieval returns poor sources the LLM tries to connect dots rather than firmly saying it doesn't know.

### Edge Cases

| Case | Result |
|------|--------|
| Empty input | Streamlit input prevents submission |
| Out-of-domain (motorcycles/boats) | ✅ Handled: said "Porsche does not manufacture" |
| Current/pricing (may be outdated) | ✅ Handled: "I don't have that specific pricing" + date note |
| No relevant sources | ✅ Handled: "I don't have information about that" |
| Duplicate sources returned | ⚠️ Happens — should deduplicate before prompt |

### Areas for Improvement

- **Cold start speed**: 224s to embed 5654 chunks via Vertex AI on first run. A pre-built index or incremental sync would make setup instant.
- **First query latency**: 62.5s due to cross-encoder model download. Pre-warming at app startup would eliminate this.
- **Retrieval depth**: Some broad technical queries don't find the most specialized article. Query expansion or fine-tuned embeddings would close this gap.
- **Source deduplication**: Same article can appear multiple times in top-k. Adding document-level dedup before the LLM prompt would improve answer diversity.
- **Response variety**: The prompt template leads to formulaic phrasing. More diverse prompt templates would make responses feel more natural.
- **Corpus expansion**: Currently 111 Wikipedia articles. Adding Porsche Newsroom press releases, owner's manuals, and spec sheets would broaden coverage significantly.
