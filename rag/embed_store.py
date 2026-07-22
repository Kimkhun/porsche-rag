import json
import math
import os
import functools
from typing import Dict, List, Tuple

import numpy as np
import chromadb
from google import genai

from .ingest import Chunk, chunk_text, _extract_date, _batch_fetch_wikipedia_dates

PROJECT_ID = "project-bc66562d-f62f-4bdd-91e"
LOCATION = "global"
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")


@functools.lru_cache(maxsize=1)
def _load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class VectorStore:
    def __init__(self):
        self.client = genai.Client(
            vertexai=True, project=PROJECT_ID, location=LOCATION
        )
        db_path = os.path.join(INDEX_DIR, "chroma")
        os.makedirs(db_path, exist_ok=True)
        self.db = chromadb.PersistentClient(path=db_path)
        self.collection = self.db.get_or_create_collection(
            name="porsche_docs",
            metadata={"hnsw:space": "cosine"},
        )
        self.doc_dates: Dict[str, str] = {}
        self._bm25_ids: List[str] = []
        self._bm25_corpus: List[List[str]] = []
        self._bm25_N: int = 0
        self._bm25_avgdl: float = 0.0
        self._bm25_df: Dict[str, int] = {}

    def build(self, chunks: List[Chunk], dates: Dict[str, str] = None) -> None:
        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {"doc_title": c.doc_title, "chunk_id": c.chunk_id, "date": (dates or {}).get(c.doc_title, "")}
            for c in chunks
        ]
        self.collection.add(embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas, ids=ids)

    def sync_with_folder(self, folder: str) -> int:
        registry = self._load_registry()
        current = {}
        loaded = 0

        filenames = sorted(os.listdir(folder))
        titles = []
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".txt", ".pdf"):
                titles.append(os.path.splitext(fn)[0].replace("_", " "))
        _batch_fetch_wikipedia_dates(titles)

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in (".txt", ".pdf"):
                continue
            path = os.path.join(folder, filename)
            mtime = os.path.getmtime(path)
            current[filename] = mtime

            if filename in registry and registry[filename] == mtime:
                continue

            title = os.path.splitext(filename)[0].replace("_", " ").title()
            text = None
            if ext == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            elif ext == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(path)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
                except Exception:
                    continue

            if not text:
                continue

            doc_date = _extract_date(filename, text or "")
            if doc_date:
                self.doc_dates[title] = doc_date
            self.collection.delete(where={"doc_title": title})
            pieces = chunk_text(text, max_words=80)
            new_chunks = [
                Chunk(chunk_id=f"{title}::{i}", doc_title=title, text=piece)
                for i, piece in enumerate(pieces)
            ]
            texts = [c.text for c in new_chunks]
            ids = [c.chunk_id for c in new_chunks]
            metadatas = [
                {"doc_title": c.doc_title, "chunk_id": c.chunk_id, "date": self.doc_dates.get(c.doc_title, "")}
                for c in new_chunks
            ]
            embeddings = self._embed(texts)
            self.collection.add(embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas, ids=ids)
            loaded += 1

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in (".txt", ".pdf"):
                continue
            title = os.path.splitext(filename)[0].replace("_", " ").title()
            if title not in self.doc_dates:
                existing = self.collection.get(where={"doc_title": title}, limit=1)
                if existing["metadatas"]:
                    dt = existing["metadatas"][0].get("date", "") or ""
                    if dt:
                        self.doc_dates[title] = dt

        if loaded > 0:
            self._rebuild_bm25()
        self._save_registry(current)
        return loaded

    def _registry_path(self) -> str:
        os.makedirs(INDEX_DIR, exist_ok=True)
        return os.path.join(INDEX_DIR, "registry.json")

    def _load_registry(self) -> dict:
        path = self._registry_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self, registry: dict) -> None:
        with open(self._registry_path(), "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    def query(self, query_text: str, top_k: int = 3, rerank: bool = True, hybrid: bool = True) -> List[Tuple[Chunk, float]]:
        fetch_k = min(top_k * 5, self.collection.count() or 1)

        if hybrid and self._bm25_N == 0:
            self._rebuild_bm25()

        if hybrid and self._bm25_N > 0:
            vec_n = max(fetch_k, top_k)
        else:
            vec_n = fetch_k

        query_vec = self._embed([query_text])
        results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=vec_n,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        out: List[Tuple[Chunk, float]] = []
        if hybrid and self._bm25_N > 0:
            query_tokens = query_text.lower().split()
            bm25_scores = self._bm25_scores(query_tokens)

            vec_scores: Dict[str, float] = {}
            for i in range(len(results["ids"][0])):
                vec_scores[results["ids"][0][i]] = 1.0 - results["distances"][0][i]

            bm25_top = np.argsort(bm25_scores)[::-1][:fetch_k]
            bm25_scores_dict: Dict[str, float] = {}
            for idx in bm25_top:
                if bm25_scores[idx] > 0:
                    bm25_scores_dict[self._bm25_ids[idx]] = float(bm25_scores[idx])

            all_ids = set(vec_scores.keys()) | set(bm25_scores_dict.keys())
            vec_ranked = sorted(vec_scores, key=lambda k: -vec_scores[k])
            bm25_ranked = sorted(bm25_scores_dict, key=lambda k: -bm25_scores_dict[k])
            vec_ranks = {cid: i + 1 for i, cid in enumerate(vec_ranked)}
            bm25_ranks = {cid: i + 1 for i, cid in enumerate(bm25_ranked)}

            K = 60
            combined = {}
            for cid in all_ids:
                score = 0.0
                if cid in vec_ranks:
                    score += 1.0 / (K + vec_ranks[cid])
                if cid in bm25_ranks:
                    score += 1.0 / (K + bm25_ranks[cid])
                combined[cid] = score

            ranked_ids = sorted(combined, key=lambda k: -combined[k])[:top_k]
            get_result = self.collection.get(ids=ranked_ids)
            id_to_idx = {cid: i for i, cid in enumerate(get_result["ids"])}
            for cid in ranked_ids:
                idx = id_to_idx.get(cid)
                if idx is None:
                    continue
                meta = get_result["metadatas"][idx]
                dt = meta.get("date", "") or ""
                if dt:
                    self.doc_dates[meta["doc_title"]] = dt
                out.append((
                    Chunk(chunk_id=meta["chunk_id"], doc_title=meta["doc_title"],
                          text=get_result["documents"][idx]),
                    float(combined[cid]),
                ))
        else:
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                dt = meta.get("date", "") or ""
                if dt:
                    self.doc_dates[meta["doc_title"]] = dt
                out.append((
                    Chunk(chunk_id=meta["chunk_id"], doc_title=meta["doc_title"],
                          text=results["documents"][0][i]),
                    float(results["distances"][0][i]),
                ))

        if rerank and len(out) > 1:
            out = self._rerank(query_text, out, top_k)
        return out[:top_k]

    def _rebuild_bm25(self) -> None:
        all_data = self.collection.get()
        self._bm25_ids = all_data["ids"]
        self._bm25_corpus = [doc.lower().split() for doc in all_data["documents"]]
        N = len(self._bm25_corpus)
        self._bm25_N = N
        self._bm25_avgdl = sum(len(d) for d in self._bm25_corpus) / max(N, 1)
        df: Dict[str, int] = {}
        for tokens in self._bm25_corpus:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self._bm25_df = df

    def _bm25_scores(self, query_tokens: List[str]) -> np.ndarray:
        k1, b = 1.5, 0.75
        N, avgdl = self._bm25_N, self._bm25_avgdl
        scores = np.zeros(len(self._bm25_corpus), dtype=np.float32)
        for i, doc_tokens in enumerate(self._bm25_corpus):
            dl = len(doc_tokens)
            if dl == 0:
                continue
            doc_counts = {}
            for t in doc_tokens:
                doc_counts[t] = doc_counts.get(t, 0) + 1
            for q in query_tokens:
                tf = doc_counts.get(q, 0)
                if tf == 0:
                    continue
                idf = math.log((N - self._bm25_df.get(q, 0) + 0.5) / (self._bm25_df.get(q, 0) + 0.5) + 1)
                scores[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        return scores

    def _rerank(self, query: str, chunks: List[Tuple[Chunk, float]], top_k: int) -> List[Tuple[Chunk, float]]:
        model = _load_reranker()
        pairs = [(query, c.text) for c, _ in chunks]
        scores = model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for (c, _), s in ranked[:top_k]]

    def _embed(self, texts: List[str]) -> np.ndarray:
        batch_size = 250
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.client.models.embed_content(
                model="gemini-embedding-001", contents=batch
            )
            all_embeddings.extend([e.values for e in result.embeddings])
        return np.array(all_embeddings, dtype=np.float32)
