import os
import re
import json
import uuid
import random
import logging
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quiz")

# ---------- Gemini setup ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Try these models in order. The first one that works is used.
# `gemini-1.5-flash` was retired on the v1beta endpoint; the new free model is
# `gemini-flash-latest` / `gemini-2.0-flash` / `gemini-2.5-flash`.
MODEL_CANDIDATES = [
    os.getenv("GEMINI_MODEL", "").strip(),
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-pro-latest",
]
MODEL_CANDIDATES = [m for m in MODEL_CANDIDATES if m]

ACTIVE_MODEL: Optional[str] = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    log.info("Gemini configured. Will probe models: %s", MODEL_CANDIDATES)
else:
    log.warning("GEMINI_API_KEY not set - AI calls will fail; fallback content used.")


def pick_model() -> str:
    """Pick the first candidate model that responds. Cached after first success."""
    global ACTIVE_MODEL
    if ACTIVE_MODEL:
        return ACTIVE_MODEL
    last_err = None
    for name in MODEL_CANDIDATES:
        try:
            m = genai.GenerativeModel(name)
            r = m.generate_content("ping", generation_config={"max_output_tokens": 5})
            _ = r.text  # force evaluate
            ACTIVE_MODEL = name
            log.info("Using Gemini model: %s", name)
            return name
        except Exception as e:
            last_err = e
            log.warning("Model %s unavailable: %s", name, e)
    raise RuntimeError(f"No Gemini model available. Last error: {last_err}")


# ---------- App ----------
app = FastAPI(title="Quiz Backend", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PDF_STORE: dict = {}
QUESTION_CACHE: dict = {}
MAX_CONTEXT_CHARS = 15000


# ---------- Models ----------
class QuestionRequest(BaseModel):
    mode: str = "subject"          # "subject" or "pdf"
    subject: Optional[str] = None
    difficulty: str = "medium"     # easy | medium | hard
    question_type: str = "mixed"
    pdf_id: Optional[str] = None
    asked: List[str] = Field(default_factory=list)


class ScoreRequest(BaseModel):
    question: dict
    user_answer: str


# ---------- Helpers ----------
def safe_parse_json(text: str):
    if not text:
        return None

    text = text.strip()

    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)

    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    return None


# A small varied fallback bank so users aren't stuck on the same fallback
FALLBACK_BANK = [
    {
        "question": "Which data structure uses Last-In-First-Out (LIFO) order?",
        "options": {"A": "Queue", "B": "Stack", "C": "Linked List", "D": "Tree"},
        "answer": "B",
        "explanation": "Stacks add/remove from the same end (LIFO).",
    },
    {
        "question": "What does CPU stand for?",
        "options": {
            "A": "Central Processing Unit",
            "B": "Computer Personal Unit",
            "C": "Central Process Utility",
            "D": "Control Processing Unit",
        },
        "answer": "A",
        "explanation": "CPU = Central Processing Unit.",
    },
    {
        "question": "HTTP status code 404 means?",
        "options": {"A": "OK", "B": "Server Error", "C": "Not Found", "D": "Redirect"},
        "answer": "C",
        "explanation": "404 indicates the requested resource was not found.",
    },
    {
        "question": "Which sorting algorithm has average time complexity O(n log n)?",
        "options": {"A": "Bubble Sort", "B": "Insertion Sort", "C": "Merge Sort", "D": "Selection Sort"},
        "answer": "C",
        "explanation": "Merge sort divides and merges in O(n log n).",
    },
    {
        "question": "Which protocol is used to send email?",
        "options": {"A": "FTP", "B": "SMTP", "C": "SNMP", "D": "HTTP"},
        "answer": "B",
        "explanation": "SMTP = Simple Mail Transfer Protocol.",
    },
    {
        "question": "In Python, which keyword defines a function?",
        "options": {"A": "func", "B": "function", "C": "def", "D": "lambda"},
        "answer": "C",
        "explanation": "`def` declares a function in Python.",
    },
    {
        "question": "What is 2^10?",
        "options": {"A": "512", "B": "1000", "C": "1024", "D": "2048"},
        "answer": "C",
        "explanation": "2^10 = 1024.",
    },
    {
        "question": "Which OSI layer handles routing?",
        "options": {"A": "Data Link", "B": "Network", "C": "Transport", "D": "Session"},
        "answer": "B",
        "explanation": "Routing is the responsibility of the Network layer (Layer 3).",
    },
]


def fallback_question(asked: List[str]) -> dict:
    asked_set = set((q or "").strip() for q in asked)
    pool = [q for q in FALLBACK_BANK if q["question"] not in asked_set]
    if not pool:
        pool = FALLBACK_BANK
    return dict(random.choice(pool))

def build_pdf_prompt(context: str, asked: List[str], question_type: str = "mixed") -> str:
    asked_block = ""

    if asked:
        bullets = "\n".join(f"- {q}" for q in asked[-15:])
        asked_block = f"\nDo NOT repeat these questions:\n{bullets}\n"

    type_instruction = {
        "mcq": "Generate ONLY MCQ questions.",
        "fill_blank": "Generate ONLY fill in the blank questions.",
        "one_word": "Generate ONLY one word answer questions.",
        "analogy": "Generate ONLY analogy questions.",
        "mixed": "Generate a MIX of MCQ, fill blank, one word, and analogy questions."
    }.get(question_type, "Generate mixed question types.")

    return f"""
You are a quiz generator.

Read the DOCUMENT and generate ONE quiz question.

{type_instruction}

Question types allowed:
- mcq
- fill_blank
- one_word
- analogy

Rules:
- Question MUST come from the DOCUMENT.
- Output ONLY valid JSON.
- No markdown.
- No explanations outside JSON.

JSON formats:

MCQ:
{{
  "type": "mcq",
  "question": "...",
  "options": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "answer": "A",
  "explanation": "..."
}}

Fill Blank:
{{
  "type": "fill_blank",
  "question": "The capital of France is ____.",
  "answer": "Paris",
  "explanation": "..."
}}

One Word:
{{
  "type": "one_word",
  "question": "What process do plants use to make food?",
  "answer": "Photosynthesis",
  "explanation": "..."
}}

Analogy:
{{
  "type": "analogy",
  "question": "Bird : Nest :: Bee : ?",
  "answer": "Hive",
  "explanation": "..."
}}

{asked_block}

DOCUMENT:
\"\"\"{context}\"\"\"
""".strip()
def build_subject_prompt(subject: str, difficulty: str, asked: List[str], question_type: str = "mixed") -> str:

    asked_block = ""

    if asked:
        bullets = "\n".join(f"- {q}" for q in asked[-20:])
        asked_block = f"\nDo NOT repeat these questions:\n{bullets}\n"

    type_instruction = {
        "mcq": "Generate ONLY MCQ questions.",
        "fill_blank": "Generate ONLY fill in the blank questions.",
        "one_word": "Generate ONLY one word answer questions.",
        "analogy": "Generate ONLY analogy questions.",
        "mixed": "Generate a MIX of MCQ, fill blank, one word, and analogy questions."
    }.get(question_type, "Generate mixed question types.")

    return f"""
You are a quiz generator.

Generate ONE quiz question.

Subject: {subject}
Difficulty: {difficulty}

{type_instruction}

Question types allowed:
- mcq
- fill_blank
- one_word
- analogy

Rules:
- Output ONLY valid JSON
- No markdown
- No explanations outside JSON

MCQ format:
{{
  "type": "mcq",
  "question": "...",
  "options": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "answer": "A",
  "explanation": "..."
}}

Fill Blank:
{{
  "type": "fill_blank",
  "question": "The capital of France is ____.",
  "answer": "Paris",
  "explanation": "..."
}}

One Word:
{{
  "type": "one_word",
  "question": "What process do plants use to make food?",
  "answer": "Photosynthesis",
  "explanation": "..."
}}

Analogy:
{{
  "type": "analogy",
  "question": "Bird : Nest :: Bee : ?",
  "answer": "Hive",
  "explanation": "..."
}}

{asked_block}
""".strip()


def call_gemini(prompt: str) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    model_name = pick_model()
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    raw = (resp.text or "").strip()
    log.info(f"GEMINI RAW RESPONSE: {raw}")
    parsed = safe_parse_json(raw)
    if not parsed:
        raise RuntimeError(f"Could not parse Gemini JSON: {raw}")
    return parsed

def normalize_question(parsed: dict) -> Optional[dict]:
    if not parsed:
        return None
    qtype = parsed.get("type", "mcq")
    # Basic validation
    if "question" not in parsed or "answer" not in parsed:
        return None
    # MCQ validation
    if qtype == "mcq":
        opts = parsed.get("options", {})
        if not all(k in opts for k in ("A", "B", "C", "D")):
            return None
    # Non-MCQ question types
    else:
        parsed["options"] = {}
    parsed.setdefault("explanation", "")
    return parsed
# ---------- Routes ----------
@app.get("/")
def root():
    return {
        "name": "Quiz Backend",
        "version": "4.0.0",
        "endpoints": ["/health", "/models", "/upload", "/question", "/score"],
    }
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai_configured": bool(GEMINI_API_KEY),
        "active_model": ACTIVE_MODEL,
        "candidates": MODEL_CANDIDATES,
        "pdfs_loaded": len(PDF_STORE),
    }
@app.get("/models")
def list_models():
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set"}
    try:
        out = []
        for m in genai.list_models():
            if "generateContent" in (m.supported_generation_methods or []):
                out.append(m.name)
        return {"models": out}
    except Exception as e:
        return {"error": str(e)}
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    try:
        data = await file.read()
        tmp_path = f"/tmp/{uuid.uuid4().hex}.pdf"
        with open(tmp_path, "wb") as f:
            f.write(data)
        reader = PdfReader(tmp_path)
        pages = len(reader.pages)
        text_parts = []
        for p in reader.pages:
            try:
                text_parts.append(p.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(text_parts).strip()
        try: os.remove(tmp_path)
        except Exception: pass
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        pdf_id = uuid.uuid4().hex
        PDF_STORE[pdf_id] = text[:200000]
        return {"pdf_id": pdf_id, "pages": pages, "chars": len(text)}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
        
@app.post("/question")
def question(req: QuestionRequest):
    cache_key = f"{req.mode}:{req.subject}:{req.difficulty}:{req.pdf_id}:{req.question_type}"
    if cache_key not in QUESTION_CACHE:
        QUESTION_CACHE[cache_key] = []
    cache = QUESTION_CACHE[cache_key]
    if cache:
        return cache.pop(0)
    asked = req.asked or []
    try:
        # ---------- PDF MODE ----------
        if req.mode == "pdf":
            text = PDF_STORE.get(req.pdf_id or "")
            if not text:
                raise HTTPException(
                    status_code=404,
                    detail="pdf_id not found."
                )
            prompt = build_pdf_prompt(
                text[:MAX_CONTEXT_CHARS],
                asked,
                req.question_type
            )
        # ---------- SUBJECT MODE ----------
        else:
            subject = (req.subject or "General Knowledge").strip()
            prompt = build_subject_prompt(
                subject,
                req.difficulty or "medium",
                asked,
                req.question_type
            )
        # ---------- AI ----------
        parsed = call_gemini(prompt)
        if not parsed:
            raise RuntimeError("Could not parse AI response")
        if isinstance(parsed, dict):
            parsed = [parsed]
        valid_questions = []
        for q in parsed:
            nq = normalize_question(q)
            if nq:
                valid_questions.append(nq)
        if not valid_questions:
            raise RuntimeError("No valid questions generated")
        QUESTION_CACHE[cache_key] = valid_questions
        return QUESTION_CACHE[cache_key].pop(0)
    except Exception as e:
        log.exception("QUESTION ERROR")
        fb = fallback_question(asked)
        fb["explanation"] = f"(AI error: {e}) " + fb["explanation"]
        return fb

@app.post("/score")
def score(req: ScoreRequest):

    qtype = req.question.get("type", "mcq")

    correct_answer = str(req.question.get("answer", "")).strip()
    user_answer = str(req.user_answer or "").strip()

    # MCQ
    if qtype == "mcq":

        is_correct = (
            user_answer.upper() == correct_answer.upper()
        )

    # Text-based answers
    else:

        is_correct = (
            user_answer.lower() == correct_answer.lower()
        )

    return {
        "correct": is_correct,
        "correct_answer": correct_answer,
        "explanation": req.question.get("explanation", ""),
    }
