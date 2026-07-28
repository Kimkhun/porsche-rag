import json
import os
import re
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str


_DATE_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "index", "wikipedia_dates.json"
)

_DATE_FILENAME_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_DATE_TEXT_PATTERNS = [
    # "As of 2024", "as of January 2024"
    re.compile(r"(?:as of|since|as at)\s+(?:(?P<month>\w+)\s+)?(?P<year>\d{4})", re.I),
    # "For the 2025 model year", "2025 model year"
    re.compile(r"(?:(?:for the|model)\s+)?(?P<year>\d{4})\s+model year", re.I),
    # "Introduced in 2023", "launched in 2023", etc.
    re.compile(r"(?:introduced|launched|released|announced|updated)\s+in\s+(?P<year>\d{4})", re.I),
    # "January 15, 2023" or "15 January 2023" (with month names)
    re.compile(r"(?P<month>\w+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"),
    re.compile(r"(?P<day>\d{1,2})\s+(?P<month>\w+)\s+(?P<year>\d{4})"),
    # "2024" standalone as a year reference in first line
    re.compile(r"^(?:The |A |An )?.*?(\d{4})", re.I),
]


def _parse_month(month_str: str) -> Optional[int]:
    return _MONTHS.get(month_str.strip().lower())


def _extract_date_from_text(text: str) -> Optional[str]:
    first_bit = text[:600]
    for pattern in _DATE_TEXT_PATTERNS:
        m = pattern.search(first_bit)
        if not m:
            continue
        d = m.groupdict()
        year_str = d.get("year")
        if not year_str:
            continue
        year = int(year_str)
        if year < 1900 or year > 2100:
            continue
        month_str = d.get("month")
        day_str = d.get("day")
        if month_str and day_str:
            month = _parse_month(month_str)
            day = int(day_str)
            if month and 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"
        if month_str:
            month = _parse_month(month_str)
            if month:
                return f"{year:04d}-{month:02d}"
        return str(year)
    return None


def _load_date_cache() -> dict:
    try:
        if os.path.exists(_DATE_CACHE_PATH):
            with open(_DATE_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_date_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_DATE_CACHE_PATH), exist_ok=True)
    with open(_DATE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _batch_fetch_wikipedia_dates(titles: List[str]) -> None:
    cache = _load_date_cache()
    needed = [t for t in titles if t not in cache]
    if not needed:
        return

    results = {}
    for i in range(0, len(needed), 50):
        batch = needed[i : i + 50]
        api = (
            "https://en.wikipedia.org/w/api.php?"
            "action=query&prop=info&titles={}&format=json&formatversion=2"
        ).format("|".join(urllib.parse.quote(t.replace(" ", "_")) for t in batch))
        try:
            with urllib.request.urlopen(api, timeout=15) as resp:
                data = json.loads(resp.read())
            pages = data.get("query", {}).get("pages", [])
            for page in pages:
                if page.get("missing"):
                    continue
                touched = page.get("touched")
                if touched:
                    results[page["title"]] = touched[:10]
        except Exception:
            pass

    cache.update(results)
    _save_date_cache(cache)


def _fetch_wikipedia_date(title: str) -> Optional[str]:
    cache = _load_date_cache()
    return cache.get(title)


def _extract_date(filename: str, text: str = "") -> Optional[str]:
    m = _DATE_FILENAME_RE.search(filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    text_date = _extract_date_from_text(text)
    if text_date:
        return text_date
    title = os.path.splitext(filename)[0].replace("_", " ")
    return _fetch_wikipedia_date(title)


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_words: int = 80, overlap_sentences: int = 2) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    prev_tail: List[str] = []
    pending_header = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        words = para.split()
        is_header = len(words) < 5
        if is_header:
            pending_header = (pending_header + " " + para).strip()
            prev_tail = []
            continue
        para = (pending_header + " " + para).strip() if pending_header else para
        pending_header = ""
        words = para.split()
        if len(words) <= max_words:
            chunk = para
            if prev_tail:
                chunk = " ".join(prev_tail) + " " + chunk
            chunks.append(chunk)
            prev_tail = _split_sentences(para)[-overlap_sentences:] if overlap_sentences else []
        else:
            sentences = _split_sentences(para)
            all_sents = list(prev_tail)
            prev_tail = []
            buf = []
            count = 0
            for s in sentences:
                s_words = s.split()
                if count + len(s_words) > max_words and buf:
                    chunk_text = " ".join(buf)
                    if all_sents:
                        chunk_text = " ".join(all_sents) + " " + chunk_text
                        all_sents = []
                    chunks.append(chunk_text)
                    buf = []
                    count = 0
                buf.append(s)
                count += len(s_words)
            if buf:
                final_chunk = " ".join(buf)
                if all_sents:
                    final_chunk = " ".join(all_sents) + " " + final_chunk
                chunks.append(final_chunk)
            prev_tail = sentences[-overlap_sentences:] if overlap_sentences else []
    if pending_header:
        chunks.append(pending_header)
    return chunks if chunks else [text]


def load_documents(folder: str) -> List[dict]:
    docs = []
    filenames = sorted(os.listdir(folder))

    titles = []
    for filename in filenames:
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".txt", ".pdf"):
            titles.append(os.path.splitext(filename)[0].replace("_", " "))
    _batch_fetch_wikipedia_dates(titles)

    for filename in filenames:
        path = os.path.join(folder, filename)
        ext = os.path.splitext(filename)[1].lower()
        title = os.path.splitext(filename)[0].replace("_", " ").title()
        text = None

        if ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        elif ext == ".pdf" and PdfReader is not None:
            try:
                reader = PdfReader(path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            except Exception:
                continue

        if text:
            doc_date = _extract_date(filename, text)
            docs.append({"title": title, "text": text, "date": doc_date})

    return docs


def build_chunk_records(docs: List[dict], chunk_size: int = 80) -> List[Chunk]:
    records = []
    for doc in docs:
        pieces = chunk_text(doc["text"], max_words=chunk_size)
        for i, piece in enumerate(pieces):
            records.append(
                Chunk(
                    chunk_id=f"{doc['title']}::{i}",
                    doc_title=doc["title"],
                    text=piece,
                )
            )
    return records
