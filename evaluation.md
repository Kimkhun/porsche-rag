# Evaluation — Porsche RAG Knowledge Base

## Test Queries

| # | Query | Type | Expected source(s) | Retrieved relevant? | Answer correct? | Notes |
|---|-------|------|-------------------|--------------------|-----------------|-------|
| 1 | Tell me about the history of Porsche | Broad overview | Porsche, Ferdinand Porsche | | | |
| 2 | What is the fastest Porsche ever made? | Comparison / opinion | Porsche 918 Spyder, Porsche 911 GT2, Porsche Taycan | | | |
| 3 | How do Porsche engines differ from other sports car engines? | Technical / comparison | Porsche flatsix engine, Porsche V8 engines, Porsche flattwelve engine | | | |
| 4 | What electric vehicles does Porsche currently produce? | Current / lineup | Porsche Taycan, Porsche Macan | | | |
| 5 | Who were the key people behind Porsche's success? | People / biography | Ferdinand Porsche, Ferry Porsche, Ferdinand Piëch, Wolfgang Porsche | | | |
| 6 | What motorsport achievements does Porsche have? | Motorsport | Porsche in motorsport, Porsche 917, Porsche 956, Porsche 919 Hybrid | | | |
| 7 | Explain the difference between the Cayenne and Macan | Comparison | Porsche Cayenne, Porsche Macan | | | |
| 8 | What is the most reliable Porsche model? | Subjective / review | (multiple possible sources) | | | |
| 9 | How has Porsche 911 design evolved over the years? | Evolution / timeline | Porsche 911, Porsche 911 (classic), Porsche 911 (996), Porsche 911 (992) | | | |
| 10 | Does Porsche manufacture motorcycles or boats? | Edge / out-of-domain | (no relevant source — should say I don't know) | | | |
| 11 | What is the price of a new Porsche 911? | Specific / current | (likely outdated or no info — system should flag source age) | | | |
| 12 | Tell me about Porsche's involvement in Formula One | Historical / motorsport | Porsche 804, Porsche in motorsport, Porsche Formula E Team | | | |

## Write-up

### Retrieval Quality

*How often did the top-3 chunks contain the right answer? Which query types worked best (broad vs specific, technical vs historical)? How did hybrid search compare to pure vector?*

### Generation Quality

*Did the LLM answer correctly when sources were strong? Did it hallucinate when sources were weak or missing? Did it properly flag insufficient or outdated information? Were the source citations clear and correct?*

### Edge Cases

- *Empty query → handled?*
- *Nonsense/out-of-domain query → handled?*
- *Follow-up question with pronoun ("what about its top speed?") → handled?*
- *Query that needs info from multiple sources → handled?*
- *Query about recent info the docs might not cover → outdated warning shown?*

### Known Limitations

*What didn't work well? What would you improve?*
