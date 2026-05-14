"""
Quiz Backend v4 - FastAPI + Google Gemini (FREE)

Modes:
  - Subject mode: pick a subject + difficulty, AI generates a question.
  - PDF mode:     upload a PDF, AI generates questions from its text.

Endpoints:
  GET  /            -> info
  GET  /health      -> health check + model
  GET  /models      -> list models the API key can access
  POST /upload      -> upload a PDF -> {pdf_id, pages, chars}
  POST /question    -> body: {mode, subject?, difficulty?, pdf_id?, asked:[...]} -> one MCQ
  POST /score       -> body: {question, user_answer} -> {correct, correct_answer, explanation}

Run locally:
  pip install -r requirements.txt
  export GEMINI_API_KEY=your_key
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

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
MAX_CONTEXT_CHARS = 15000


# ---------- Models ----------
class QuestionRequest(BaseModel):
    mode: str = "subject"          # "subject" or "pdf"
    subject: Optional[str] = None
    difficulty: str = "medium"     # easy | medium | hard
    pdf_id: Optional[str] = None
    asked: List[str] = Field(default_factory=list)


class ScoreRequest(BaseModel):
    question: dict
    user_answer: str


# ---------- Helpers ----------
def safe_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
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


def build_pdf_prompt(context: str, asked: List[str]) -> str:
    asked_block = ""
    if asked:
        bullets = "\n".join(f"- {q}" for q in asked[-15:])
        asked_block = f"\nDo NOT repeat or paraphrase these:\n{bullets}\n"
    return f"""You are a quiz generator. Read the DOCUMENT and create ONE multiple-choice question.

Rules:
- Question must be answerable using only the DOCUMENT.
- Provide exactly 4 options labeled A, B, C, D. Exactly one correct.
- Output ONLY a JSON object, no prose, no markdown.

JSON shape:
{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","explanation":"..."}}
{asked_block}
DOCUMENT:
\"\"\"{context}\"\"\"
""".strip()


def build_subject_prompt(subject: str, difficulty: str, asked: List[str]) -> str:
    asked_block = ""
    if asked:
        bullets = "\n".join(f"- {q}" for q in asked[-20:])
        asked_block = f"\nDo NOT repeat or paraphrase these previously asked questions:\n{bullets}\n"
    seed = random.randint(1000, 999999)
    return f"""You are a quiz generator. Create ONE multiple-choice question.

Subject: {subject}
Difficulty: {difficulty}
Random seed (use it to vary topic/angle): {seed}

Rules:
- Provide exactly 4 options labeled A, B, C, D. Exactly one correct.
- Make the question UNIQUE; vary subtopics each time.
- Output ONLY a JSON object, no prose, no markdown.

JSON shape:
{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","explanation":"short reason"}}
{asked_block}""".strip()


def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    model_name = pick_model()
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
    prompt,
    generation_config={
        "temperature": 0.9,
        "max_output_tokens": 600,
    },
    )
    return (resp.text or "").strip()


def normalize_question(parsed: dict) -> Optional[dict]:
    if not parsed or "question" not in parsed or "options" not in parsed or "answer" not in parsed:
        return None
    opts = parsed.get("options", {})
    if not all(k in opts for k in ("A", "B", "C", "D")):
        return None
    ans = str(parsed.get("answer", "A")).strip().upper()
    if ans not in ("A", "B", "C", "D"):
        ans = "A"
    parsed["answer"] = ans
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
    asked = req.asked or []

    if req.mode == "pdf":
        text = PDF_STORE.get(req.pdf_id or "")
        if not text:
            raise HTTPException(status_code=404, detail="pdf_id not found. Upload again.")
        prompt = build_pdf_prompt(text[:MAX_CONTEXT_CHARS], asked)
    else:
        subject = (req.subject or "General Knowledge").strip()
        prompt = build_subject_prompt(subject, req.difficulty or "medium", asked)

    if not GEMINI_API_KEY:
        return fallback_question(asked)

    # Try AI up to 2 times
    last_err = None
    for _ in range(2):
        try:
            raw = call_gemini(prompt)
            parsed = normalize_question(safe_parse_json(raw))
            if parsed and parsed["question"].strip() not in set(asked):
                return parsed
        except Exception as e:
            last_err = e
            log.warning("Gemini error: %s", e)
            break

    fb = fallback_question(asked)
    if last_err:
        fb["explanation"] = f"(AI error: {last_err}) " + fb["explanation"]
    return fb


@app.post("/score")
def score(req: ScoreRequest):
    correct = str(req.question.get("answer", "")).strip().upper()
    user = str(req.user_answer or "").strip().upper()
    return {
        "correct": user == correct and correct in ("A", "B", "C", "D"),
        "correct_answer": correct,
        "explanation": req.question.get("explanation", ""),
    }
