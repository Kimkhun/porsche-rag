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
        "Rewrite the user's question into a search query that will find the most relevant articles. "
        "Rules:\n"
        "- Include the core topic AND add related keywords, model names, or synonyms that help retrieval.\n"
        "- Disambiguate pronouns like 'it', 'they', 'that' using conversation context.\n"
        "- Expand broad categories with known specific examples from that category.\n"
        "- For specific model questions, include the model name and related terms.\n"
        "- For comparison questions, include BOTH items being compared.\n"
        "- For technical questions, include the key technical terms.\n"
        "Examples:\n"
        "  'What EVs does Porsche make?' -> 'Taycan Macan EV Mission E electric vehicle hybrid Panamera'\n"
        "  'Tell me about the 911' -> 'Porsche 911 930 964 993 996 997 991 992 classic generations'\n"
        "  'Cayenne vs Macan difference' -> 'Cayenne Macan comparison SUV difference size engine'\n"
        "  'Who designed the 911' -> '911 design Butzi Porsche Ferdinand Alexander Porsche designer'\n"
        "  'Fastest Porsche' -> 'fastest top speed 918 Spyder 919 Hybrid 911 GT2 RS performance'\n"
        "  'How do engines work' -> 'flat-six boxer V8 V10 engine motor technology layout'\n"
        "  'Porsche Le Mans wins' -> 'Le Mans 917 956 962 919 Hybrid endurance racing victory'\n"
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
