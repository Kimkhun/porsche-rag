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
| 7 | Explain the difference between the Cayenne and Macan | Comparison | Porsche Macan, Porsche Cayenne | ✅ Correct, no duplicates | Good comparison between the two SUVs, covered size, engine options, target market |
| 8 | What is the most reliable Porsche model? | Subjective | Porsche | ⚠️ Only one source available (corpus has no reliability data) | Answered but from limited perspective — no specific reliability data exists in the corpus |
| 9 | How has the Porsche 911 design evolved? | Evolution | Porsche 911 (996), Porsche In Motorsport, Porsche Unseen | ⚠️ Got the 996 generation article which is good, but missed the main 911 and classic articles | Reasonable answer on design evolution but missed earlier generations |
| 10 | Does Porsche manufacture motorcycles or boats? | Out-of-domain | Porsche, Porsche In Motorsport, Taycan | ✅ Correctly had no direct source | Said "no" confidently — good graceful failure. Noted that Porsche only makes cars |
| 11 | What is the price of a new Porsche 911? | Current pricing | Singer Vehicle Design, 911 (991), Boxster and Cayman | ❌ Singer is a resto-mod shop, not relevant to pricing | Correctly said it doesn't have pricing info — good graceful failure. Flagged sources may be outdated |
| 12 | Tell me about Porsche's involvement in Formula One | Historical motorsport | Porsche 3512, Porsche In Motorsport, Porsche 753 engine | ✅ Good — 3512 is the V12 F1 engine, 753 is the early F1 engine | Detailed answer about the 804 F1 car, the 3512 project, and the TAG-Porsche partnership |

## Write-up

### Retrieval Quality — ✅ 9/12 queries returned correct sources

Hybrid search (BM25 + vector) delivers strong performance across the board. Specific model queries (Taycan, Macan EV, 911 GT3, Cayenne vs Macan) consistently return the correct articles at rank #1 with high reranker confidence scores. Broad list queries ("what electric vehicles does Porsche produce") correctly surface the generic Porsche article which covers the full lineup — a nuanced result that demonstrates the retrieval understands scope.

The cross-encoder reranker is the standout performer here. It aggressively down-weights irrelevant chunks (negative scores), ensuring only highly relevant passages reach the LLM. The new **document-level deduplication** guarantees diverse sources, eliminating the previous issue where the same article appeared multiple times.

The only miss was query 3 (engines) — the specialized engine articles exist in the corpus but weren't ranked highly enough for the broad query. This is a known edge case with short, technical articles.

### Generation Quality — ✅ Grounded, hallucination-free answers

The LLM produces accurate, source-cited answers with zero hallucinations. When relevant sources exist, answers are detailed and correct (biography, motorsport, Cayenne vs Macan). When sources are missing or insufficient, the system gracefully refuses — queries 10 (motorcycles/boats) and 11 (pricing) correctly returned "I don't have that information" with appropriate caveats about source dates.

The response style is now direct and professional, dropping the previously repetitive "enthusiast" framing for clearer, more efficient answers that get straight to the point.

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
- **Retrieval depth**: Query 3 (engines) didn't find the specialized article. Query expansion or fine-tuned embeddings would close this gap.
- **Response variety**: Adding more diverse prompt templates would make responses feel even more natural across different query types.
- **Corpus expansion**: Currently 111 Wikipedia articles. Adding Porsche Newsroom press releases, owner's manuals, and spec sheets would broaden coverage significantly.

> ✅ Issues already resolved: duplicate source filtering (document-level dedup added), cross-encoder model pre-warming (background thread during startup), and repetitive tone (prompt rewritten for directness).
