from datetime import date
from typing import Dict, Generator, List, Tuple

from google import genai

from .ingest import Chunk

PROJECT_ID = "project-bc66562d-f62f-4bdd-91e"
LOCATION = "asia-southeast1"
MAX_HISTORY = 999

_CLIENT = None

def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return _CLIENT


def _format_history(history: List[dict]) -> str:
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for msg in history[-MAX_HISTORY:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_llm_prompt(query: str, retrieved: List[Tuple[Chunk, float]], history: List[dict] = None, doc_dates: Dict[str, str] = None) -> str:
    today = date.today().isoformat()
    source_lines = []
    for c, _ in retrieved:
        cd = (doc_dates or {}).get(c.doc_title)
        date_tag = f" [{cd}]" if cd else ""
        source_lines.append(f"Source: {c.doc_title}{date_tag}\n{c.text}")
    context = "\n\n".join(source_lines)
    history_block = _format_history(history or [])
    prompt_parts = [
        f"Today's date: {today}\n",
        "You are a Porsche expert. Answer clearly, directly, and cite which source(s) you used. Be concise — no fluff or forced enthusiasm.",
        "RULES:",
        "1. Base your answer on the sources. If the sources have related but not exact info, use it — connect the dots and note what you're inferring.",
        "2. If the sources truly have nothing relevant, say you don't have info on that specific topic.",
        "3. If a source has a date and the info may be outdated (e.g. asking about a 2025 model but source is from 2023), mention the date and note it might have changed.",
        "4. If a source has no date and the question asks about recent releases, note the source age is unknown.",
        "5. Answer from the perspective of the source's date — describe what was true when it was written.\n",
    ]
    if history_block:
        prompt_parts.append(history_block + "\n")
    prompt_parts.append(f"Current sources:\n{context}\n")
    prompt_parts.append(f"Question: {query}\nAnswer:")
    return "\n".join(prompt_parts)


def rewrite_search_query(raw_query: str, history: List[dict] = None) -> str:
    client = _get_client()
    hist_block = _format_history(history or [])
    prompt = (
        "You are a search query optimizer for a Porsche knowledge base. "
        "Your job is to rewrite the user's question into the MOST comprehensive, detailed search query possible. "
        "The rewritten query will be matched against article titles and text using both keyword (BM25) and semantic (vector) search. "
        "Rules:\n"
        "- Include the original question's key terms AND add every related term, model name, synonym, and concept you can think of.\n"
        "- Disambiguate pronouns like 'it', 'they', 'that' using conversation context.\n"
        "- For broad topics: list every model, person, or concept that belongs to that category.\n"
        "- For specific models: include the model code, generation names, and related technical terms.\n"
        "- For comparisons: include both items being compared plus the comparison dimensions.\n"
        "- For people: include their full name, role, and what they're known for.\n"
        "- For technical topics: include all related engineering terms.\n"
        "Examples:\n"
        "  'What EVs does Porsche make?' -> 'Taycan Macan EV Mission E electric vehicle battery hybrid Panamera SE e-hybrid 800V charging J1 platform PPE'\n"
        "  'Tell me about the 911' -> 'Porsche 911 930 964 993 996 997 991 992 generations classic evolution design engine rear-engine sports car Carrera Turbo GT3 GT2 RS'\n"
        "  'Cayenne vs Macan difference' -> 'Cayenne Macan comparison difference size engine SUV luxury performance off-road towing capacity price'\n"
        "  'Who designed the 911' -> 'Ferdinand Alexander Porsche Butzi 911 design designer styling creator origin story'\n"
        "  'Fastest Porsche' -> 'top speed fastest quickest 918 Spyder 919 Hybrid 911 GT2 RS performance acceleration Nurburgring lap time horsepower'\n"
        "  'Porsche engines' -> 'engine motor flat-six boxer V8 V10 V12 turbo naturally aspirated water-cooled air-cooled Mezger configuration power'\n"
        "  'History of Porsche' -> 'history founding Ferdinand Porsche origins 1931 Volkswagen Beetle Gmünd 356 Stuttgart company timeline evolution'\n"
        "  'Le Mans wins' -> 'Le Mans 24 Hours endurance race victory 917 956 962 919 Hybrid overall win motorsport championship'\n"
        "- Output ONLY the rewritten query, nothing else.\n\n"
    )
    if hist_block:
        prompt += hist_block + "\n\n"
    prompt += f"Latest question: {raw_query}\nRewritten query:"
    response = client.models.generate_content(model="gemini-3.5-flash", contents=prompt)
    return response.text.strip()


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "No relevant passages were found for that query."
    lines = [f"Top passages related to: \u201c{query}\u201d\n"]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.doc_title}, score={score:.2f}] {chunk.text}\n")
    return "\n".join(lines)


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]], history: List[dict] = None, doc_dates: Dict[str, str] = None) -> str:
    if not retrieved:
        return "No relevant passages were found to answer that query."
    prompt = _build_llm_prompt(query, retrieved, history, doc_dates)
    client = _get_client()
    response = client.models.generate_content(model="gemini-3.5-flash", contents=prompt)
    return response.text


def llm_answer_stream(query: str, retrieved: List[Tuple[Chunk, float]], history: List[dict] = None, doc_dates: Dict[str, str] = None) -> Generator[str, None, None]:
    if not retrieved:
        yield "No relevant passages were found to answer that query."
        return
    prompt = _build_llm_prompt(query, retrieved, history, doc_dates)
    client = _get_client()
    stream = client.models.generate_content_stream(model="gemini-3.5-flash", contents=prompt)
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive", history: List[dict] = None) -> str:
    if mode == "llm":
        return llm_answer(query, retrieved, history)
    return extractive_answer(query, retrieved)
