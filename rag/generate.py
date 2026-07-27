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
        "You are a search query optimizer for a Porsche knowledge base with 111 Wikipedia articles covering models, history, people, technology, and motorsport. "
        "Your job is to rewrite the user's question into the MOST comprehensive, exhaustive search query possible. "
        "Every term you add increases the chance BM25 keyword search finds the right articles. "
        "Rules:\n"
        "- Include the original terms AND every related model name, code name, generation number, technical term, person name, year, synonym, and concept.\n"
        "- Disambiguate pronouns like 'it', 'they', 'that' using conversation context.\n"
        "- For broad topics: list ALL models, people, concepts, and terms in that category.\n"
        "- For specific models: include all generation codes, variants, engine types, years, and related models.\n"
        "- For comparisons: include both items, their differences, categories, and shared terms.\n"
        "- For people: include full name, nickname, role, contributions, and associated models.\n"
        "- For technical topics: include every related engineering term, system, and component.\n"
        "- For history: include era names, years, locations, people, and key events.\n"
        "- For motorsport: include series names, cars, drivers, teams, years, tracks, championships.\n"
        "- For design: include designer names, design elements, philosophy, generations.\n"
        "Generate the most exhaustive query possible — 50 to 200+ terms is normal.\n"
        "Examples:\n"
        "  'What electric vehicles does Porsche make?' -> 'Taycan Cross Turismo Sport Turismo Turbo Turbo S 4S base RWD sedan estate shooting brake Macan EV electric 718 Boxster Cayman EV Mission E Mission R Mission X Cross Turismo concept battery electric vehicle BEV hybrid plug-in hybrid PHEV Panamera S E-Hybrid 4 E-Hybrid Turbo S E-Hybrid Cayenne E-Hybrid Coupe 911 T-Hybrid Carrera GTS electrified J1 platform PPE Premium Platform Electric SSP electric motor PSM permanent magnet synchronous battery pack 800V 800-volt lithium-ion NCM prismatic cells recuperation regenerative braking charging AC DC fast charging range WLTP EPA electricity zero-emission sustainable e-mobility e-performance Porsche Electric strategy carbon neutrality 2025 2030'\n"
        "  'Tell me about the 911' -> 'Porsche 911 901 original classic 930 964 993 996 997 991 992 generations evolution history heritage icon sports car rear-engine rear-engined flat-six boxer air-cooled water-cooled Carrera Carrera S Carrera 4S GTS Turbo Turbo S GT3 GT3 RS GT2 GT2 RS Targa Targa 4 Targa 4S Cabriolet Speedster S/T Sport Classic Dakar rally Safari off-road Mezger engine DFI direct fuel injection VTG variable turbine geometry PDK dual-clutch transmission 7-speed 8-speed manual 7-speed rear-wheel drive all-wheel drive PASM active suspension torque vectoring PTV rear-axle steering carbon ceramic brakes PCCB lightweight aluminum steel chassis production 1963 1964 1965 2024 2025 anniversary limited edition special model Weissach package Clubsport package wing spoiler aerodynamics coupe convertible grand tourer motorsport racing heritage iconic silhouette design evolution Ferdinand Alexander Porsche Butzi HVA Hermann Valentin design philosophy timeless classic modern reinterpretation luxury sports car benchmark performance driver-focused dynamics handling steering feedback Nurburgring lap time top speed acceleration horsepower hp PS kw torque Nm lb-ft displacement 3.0 3.8 4.0 liter naturally aspirated twin-turbocharged intercooler'\n"
        "  'Cayenne vs Macan difference' -> 'Cayenne Macan comparison difference vs versus compare dimensions size length width height wheelbase weight curb weight towing capacity towing payload ground clearance approach angle departure angle cargo space trunk liters cubic feet passenger seating legroom headroom interior design dashboard infotainment PCM Porsche Communication Management engine V6 V8 turbo diesel diesel S GTS Turbo Turbo S E-Hybrid Coupe coupe SUV sport utility vehicle crossover luxury midsize full-size three-row two-row seating capacity price cost MSRP fuel economy MPG efficiency performance acceleration 0-60 0-100 quarter-mile top speed handling agility off-road capability on-road comfort towing boat trailer family practicality daily driver road trip highway transmission Tiptronic automatic PDK dual-clutch AWD all-wheel drive platform MLB Volkswagen Group SUV Leipzig production manufacturing sales popularity market position target buyer'\n"
        "  'Who designed the 911' -> 'Ferdinand Alexander Porsche Butzi Ferdinand Alexander Butzi Porsche designer stylist creator 911 design story origin history Type 901 1963 original concept sketch clay model silhouette shape iconic timeless design philosophy form follows function pure functionalism Erwin Komenda body designer coachbuilder chief engineer Reutter body construction Heinrich Klie design team styling studio Weissach Stuttgart Zuffenhausen Hamburg influence inspiration Beetle 356 lineage evolution classic modern reinterpretation legacy grandson Ferry Porsche family dynasty'\n"
        "  'Fastest Porsche ever made' -> 'fastest quickest top speed maximum velocity acceleration 0-60 0-100 0-200 0-300 km/h mph quarter-mile standing kilometer Nurburgring Nordschleife lap record production car street legal supercar hypercar performance benchmark horsepower hp PS kw torque Nm lb-ft engine power weight-to-power ratio downforce aerodynamics drag coefficient 918 Spyder 919 Hybrid Evo 919 Hybrid 911 GT2 RS 911 GT3 RS Carrera GT Taycan Turbo GT Weissach package 911 Turbo S 935 917 956 962 963 LMP1 Group C Can-Am sports car racing fastest accelerating most powerful highest top speed track weapon street legal race car limited edition special model record holder time attack World Record'\n"
        "  'How do Porsche engines work' -> 'Porsche engine motor flat-six boxer horizontally opposed V8 V10 V12 naturally aspirated turbocharged twin-turbo Bi-Turbo VTG variable turbine geometry Mezger engine DFI direct fuel injection FSI stratified injection water-cooled air-cooled dry-sump lubrication wet-sump lubrication 3.0 3.2 3.4 3.6 3.8 4.0 4.5 5.0 liter displacement bore stroke compression ratio connecting rods pistons crankshaft cylinder head valvetrain DOHC VarioCam VarioCam Plus VarioRam variable valve timing variable intake manifold intake exhaust manifold catalytic converter emissions horsepower hp PS kw torque Nm lb-ft redline rpm rev-limit flat-plane crank cross-plane crank configuration layout design engineering technology history evolution Carrera GT V10 naturally aspirated V10 V12 Formula One F1 TAG-Porsche 917 512 flat-12 908 356 four-cam engine four-cam roller bearing crank 550 Spyder Fuhrmann engine race engine motorsport engine development Weissach engine shop prototype concept sound exhaust note induction noise throttle response turbo lag power curve delivery powerband heat management cooling oil temperature intercooler charge air cooler intercooler efficiency'\n"
        "  'History of Porsche' -> 'history founding origins Ferdinand Porsche founder 1931 engineering office Stuttgart Dr. Ing. h.c. F. Porsche GmbH Volkswagen Beetle Type 1 Auto Union Grand Prix car Cisitalia Grand Prix 356 first sports car Gmünd Austria production 1948 356 SL 356 America 356 Carrera 356 Speedster 356 Convertible 550 Spyder Fuhrmann engine Carrera four-cam 718 RSK Formula One 804 Formula 1 1962 917 956 962 Le Mans 24 Hours overall victories 1970 1971 1976 1977 1979 1981 1982 1983 1984 1985 1986 1987 1994 1996 1997 1998 2015 2016 2017 Type 64 Berlin-Rome 1939 Porsche family Ferry Porsche son Ferdinand Alexander Porsche grandson Wolfgang Porsche board management timelines company history Porsche AG Porsche Holding GmbH Stuttgart Zuffenhausen Leipzig manufacturing plant milestones anniversary 75 years 50 years heritage evolution timeline chronological corporate history financial public offering IPO SE Volkswagen Group merger story key figures Wendelin Wiedeking Matthias Müller Oliver Blume'\n"
        "  'Porsche Le Mans victories' -> 'Le Mans 24 Heures 24 Hours endurance race Circuit de la Sarthe Sarthe France overall victory winners list record most wins 19 overall victories 917 KH 917 LH short-tail long-tail 1970 Hans Herrmann Richard Attwood 1971 Helmut Marko Gijs van Lennep Martini Porsche 936 1976 Jacky Ickx Gijs van Lennep 1977 Jacky Ickx Henri Pescarolo Mickey Thompson 1979 Bill Whittington Don Whittington Klaus Ludwig Kremer 935 K3 956 1982 Jacky Ickx Derek Bell 1983 Al Holbert Hurley Haywood Vern Schuppan 1984 Klaus Ludwig Henri Pescarolo 1985 Klaus Ludwig Paolo Barilla John Winter John Fitzpatrick 962 1986 Derek Bell Hans-Joachim Stuck Al Unser Jr. 1987 Derek Bell Hans-Joachim Stuck Al Holbert Dauer 962 Le Mans 1994 Yannick Dalmas Hurley Haywood Mauro Baldi 911 GT1 1998 Laurent Aïello Stéphane Ortelli Allan McNish WSC-95 1996 Alexander Wurz Davy Jones Manuel Reuter WSC-95 Joest 1997 Michele Alboreto Stefan Johansson Tom Kristensen 919 Hybrid LMP1 2015 Earl Bamber Nico Hülkenberg Nick Tandy 2016 Romain Dumas Marc Lieb Neel Jani 2017 Earl Bamber Timo Bernhard Brendon Hartley Porsche Motorsport factory team privateer Joest Racing John Wyer Racing Gulf Penske Martini Rothmans Vaillant Salzburg history statistics records heritage greatest manufacturer world domination 1970s dominant Group C LMP1 era 2023 2024 963 LMDh GTP comeback future'\n"
        "Output ONLY the rewritten query, nothing else.\n\n"
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
