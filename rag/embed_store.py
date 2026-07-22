import json
import math
import os
import re
import functools
import time
import urllib.request
import urllib.parse
from typing import Dict, List, Tuple, Optional

import numpy as np
import chromadb
from google import genai

from .ingest import Chunk, chunk_text, _extract_date, _batch_fetch_wikipedia_dates

PROJECT_ID = "project-bc66562d-f62f-4bdd-91e"
LOCATION = "asia-southeast1"
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")

CORE_WIKIPEDIA_TITLES = [
    "Car Engineer of the Century", "Ernst Fuhrmann", "Erwin Komenda", "European delivery",
    "Ferdinand Alexander Porsche", "Ferdinand Piëch", "Ferdinand Porsche", "Ferry Porsche",
    "Gemballa", "List of Porsche concept vehicles", "List of Porsche vehicles", "Lohner-Porsche",
    "Louise Piëch", "Michael Mauer", "Need for Speed Porsche Unleashed", "Porsche",
    "Porsche 3512", "Porsche 356", "Porsche 3561", "Porsche 3562", "Porsche 356SL",
    "Porsche 360", "Porsche 547 engine", "Porsche 550", "Porsche 597", "Porsche 64",
    "Porsche 645", "Porsche 718", "Porsche 718 Boxster and Cayman 982", "Porsche 753 engine",
    "Porsche 804", "Porsche 901", "Porsche 904", "Porsche 906", "Porsche 907", "Porsche 908",
    "Porsche 909 Bergspyder", "Porsche 910", "Porsche 911", "Porsche 911 GT1",
    "Porsche 911 GT2", "Porsche 911 GT3", "Porsche 911 RSR", "Porsche 911 classic",
    "Porsche 912", "Porsche 914-6 GT", "Porsche 914", "Porsche 917", "Porsche 918 Spyder",
    "Porsche 919 Hybrid", "Porsche 924", "Porsche 928", "Porsche 930", "Porsche 934",
    "Porsche 935", "Porsche 936", "Porsche 944", "Porsche 953", "Porsche 959",
    "Porsche 961", "Porsche 962", "Porsche 963", "Porsche 964", "Porsche 968",
    "Porsche 984", "Porsche 989", "Porsche 991", "Porsche 992", "Porsche 993",
    "Porsche 996", "Porsche 997", "Porsche B32", "Porsche Boxster 986", "Porsche Boxster and Cayman",
    "Porsche Boxster and Cayman 981", "Porsche Boxster and Cayman 987", "Porsche C88",
    "Porsche Carrera Cup", "Porsche Carrera Cup Germany", "Porsche Carrera GT",
    "Porsche Cayenne", "Porsche Cayman", "Porsche Cayman GT4", "Porsche Design",
    "Porsche Design Tower Stuttgart", "Porsche Design Tower Sunny Isles Beach",
    "Porsche Engineering", "Porsche Formula E Team", "Porsche Holding", "Porsche Junior",
    "Porsche LMP1-98", "Porsche Macan", "Porsche Mission E", "Porsche Museum",
    "Porsche Panamera", "Porsche RS Spyder", "Porsche Rennsport Reunion", "Porsche SE",
    "Porsche Supercup", "Porsche Taycan", "Porsche Unseen", "Porsche V10 engine",
    "Porsche V8 engines", "Porsche VIN specification", "Porsche Vision 357",
    "Porsche Vision 357 Speedster", "Porsche Vision Gran Turismo", "Porsche WSC95",
    "Porsche family", "Porsche flateight engines", "Porsche flatsix engine",
    "Porsche flattwelve engine", "Porsche in motorsport", "Porsche type numbers",
    "Ruf Automobile", "Singer Vehicle Design", "VarioCam", "VarioRam",
    "Volkswagen Beetle", "Wendelin Wiedeking", "Wolfgang Porsche"
]


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
        self._bm25_inverted: Dict[str, List[int]] = {}

    def build(self, chunks: List[Chunk], dates: Dict[str, str] = None) -> None:
        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {"doc_title": c.doc_title, "chunk_id": c.chunk_id, "date": (dates or {}).get(c.doc_title, "")}
            for c in chunks
        ]
        self.collection.add(embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas, ids=ids)

    def _batch_fetch_wikipedia_content(self, titles: List[str]) -> Dict[str, Tuple[str, str]]:
        results = {}

        for i in range(0, len(titles), 20):
            batch = titles[i : i + 20]
            titles_str = "|".join(urllib.parse.quote(t) for t in batch)
            api = (
                "https://en.wikipedia.org/w/api.php?"
                "action=query&prop=revisions|info&rvprop=content&titles={}&redirects=1&format=json&formatversion=2"
            ).format(titles_str)
            try:
                req = urllib.request.Request(api, headers={"User-Agent": "PorscheRAG/1.0 (student project)"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                for page in data.get("query", {}).get("pages", []):
                    if page.get("missing"):
                        continue
                    resolved = page.get("title")
                    touched = page.get("touched", "")[:10]
                    revisions = page.get("revisions", [])
                    if not revisions:
                        continue
                    wikitext = revisions[0].get("content", "")
                    # Strip wiki markup to plain text
                    text = wikitext
                    text = re.sub(r"'''?|'''?", "", text)
                    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
                    text = re.sub(r"\{\{[^}]*\}\}", "", text)
                    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
                    text = re.sub(r"<[^>]+>", "", text)
                    text = re.sub(r"={2,}\s*([^=]+)\s*={2,}", r"\n\n\1\n\n", text)
                    text = re.sub(r"\n{3,}", "\n\n", text).strip()
                    if text:
                        results[resolved] = (text, touched)
            except Exception:
                pass
            time.sleep(1.5)

        return results

    def sync_with_folder(self, folder: str) -> int:
        registry = self._load_registry()
        current = {}
        loaded = 0

        # Step 1: Query ChromaDB to see what is already indexed
        try:
            existing_data = self.collection.get()
            indexed_titles = {
                meta["doc_title"] 
                for meta in existing_data.get("metadatas", []) 
                if meta and "doc_title" in meta
            }
        except Exception:
            indexed_titles = set()

        # Step 2: Fetch any missing Wikipedia pages from the API in batches
        missing_titles = []
        for title in CORE_WIKIPEDIA_TITLES:
            normalized_title = title.title()
            if normalized_title in indexed_titles:
                # Also restore doc_dates cache from metadata
                if normalized_title not in self.doc_dates:
                    try:
                        matching = self.collection.get(where={"doc_title": normalized_title}, limit=1)
                        if matching["metadatas"]:
                            dt = matching["metadatas"][0].get("date", "")
                            if dt:
                                self.doc_dates[normalized_title] = dt
                    except Exception:
                        pass
                continue
            missing_titles.append(title)

        if missing_titles:
            batch_results = self._batch_fetch_wikipedia_content(missing_titles)
            all_texts = []
            all_ids = []
            all_metadatas = []
            for title, (text, touched) in batch_results.items():
                normalized_title = title.title()
                if touched:
                    self.doc_dates[normalized_title] = touched
                pieces = chunk_text(text, max_words=80)
                for i, piece in enumerate(pieces):
                    all_texts.append(piece)
                    chunk_id = f"{normalized_title}::{i}"
                    all_ids.append(chunk_id)
                    all_metadatas.append({"doc_title": normalized_title, "chunk_id": chunk_id, "date": self.doc_dates.get(normalized_title, "")})
            if all_texts:
                embeddings = self._embed(all_texts)
                self.collection.add(embeddings=embeddings.tolist(), documents=all_texts, metadatas=all_metadatas, ids=all_ids)
                loaded = len(batch_results)

        # Step 3: Sync local files from target directory
        if os.path.exists(folder):
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
        fetch_k = min(top_k * 2, self.collection.count() or 1)

        if hybrid and self._bm25_N == 0:
            self._rebuild_bm25()

        if hybrid and self._bm25_N > 0:
            vec_n = max(fetch_k, top_k)
        else:
            vec_n = fetch_k

        t_start = time.perf_counter()
        query_vec = self._embed([query_text])
        t_after_embed = time.perf_counter()
        
        results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=vec_n,
        )
        t_after_chroma = time.perf_counter()

        if not results["ids"] or not results["ids"][0]:
            self.last_query_timings = {
                "embed": t_after_embed - t_start,
                "chroma": t_after_chroma - t_after_embed,
                "bm25": 0.0,
                "rerank": 0.0,
                "total": t_after_chroma - t_start
            }
            return []

        out: List[Tuple[Chunk, float]] = []
        t_bm25_start = time.perf_counter()
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
        t_bm25_end = time.perf_counter()

        t_rerank_start = time.perf_counter()
        rerank_active = False
        if rerank and len(out) > 1:
            out = self._rerank(query_text, out, top_k)
            rerank_active = True
        t_rerank_end = time.perf_counter()
        
        self.last_query_timings = {
            "embed": t_after_embed - t_start,
            "chroma": t_after_chroma - t_after_embed,
            "bm25": t_bm25_end - t_bm25_start if (hybrid and self._bm25_N > 0) else 0.0,
            "rerank": t_rerank_end - t_rerank_start if rerank_active else 0.0,
            "total": time.perf_counter() - t_start
        }
        return out[:top_k]

    def _rebuild_bm25(self) -> None:
        all_data = self.collection.get()
        self._bm25_ids = all_data["ids"]
        self._bm25_corpus = [doc.lower().split() for doc in all_data["documents"]]
        N = len(self._bm25_corpus)
        self._bm25_N = N
        self._bm25_avgdl = sum(len(d) for d in self._bm25_corpus) / max(N, 1)
        df: Dict[str, int] = {}
        inverted: Dict[str, List[int]] = {}
        for i, tokens in enumerate(self._bm25_corpus):
            seen = set()
            for t in tokens:
                if t not in seen:
                    seen.add(t)
                    df[t] = df.get(t, 0) + 1
                    if t not in inverted:
                        inverted[t] = []
                    inverted[t].append(i)
        self._bm25_df = df
        self._bm25_inverted = inverted

    def _bm25_scores(self, query_tokens: List[str]) -> np.ndarray:
        k1, b = 1.5, 0.75
        N, avgdl = self._bm25_N, self._bm25_avgdl
        scores = np.zeros(len(self._bm25_corpus), dtype=np.float32)
        candidates: set = set()
        for q in query_tokens:
            if q in self._bm25_inverted:
                candidates.update(self._bm25_inverted[q])
        if not candidates:
            return scores
        for i in candidates:
            doc_tokens = self._bm25_corpus[i]
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
