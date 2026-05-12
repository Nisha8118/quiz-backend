"""
Quiz Backend - FastAPI + Google Gemini (FREE)

Endpoints:
  GET  /            -> info
  GET  /health      -> health check (also reports if Gemini key is configured)
  POST /upload      -> upload a PDF, returns {pdf_id, pages, chars}
  POST /question    -> body: {pdf_id, asked: [..]} -> returns one MCQ
  POST /score       -> body: {question, user_answer} -> returns {correct, correct_answer, explanation}

Run locally:
  pip install -r requirements.txt
  export GEMINI_API_KEY=your_key
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import re
import json
import uuid
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
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    log.info("Gemini configured with model: %s", MODEL_NAME)
else:
    log.warning("GEMINI_API_KEY not set - /question and /score will use fallback content.")

# ---------- App ----------
app = FastAPI(title="Quiz Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory PDF store: {pdf_id: text}
PDF_STORE: dict = {}
MAX_CONTEXT_CHARS = 15000  # keep prompt small to avoid quota issues


# ---------- Models ----------
class QuestionRequest(BaseModel):
    pdf_id: str
    asked: List[str] = Field(default_factory=list)


class ScoreRequest(BaseModel):
    question: dict
    user_answer: str


# ---------- Helpers ----------
def safe_parse_json(text: str) -> Optional[dict]:
    """Extract a JSON object from a model response that may include markdown fences or prose."""
    if not text:
        return None
    # Strip code fences
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    # Try direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def fallback_question() -> dict:
    return {
        "question": "Which of the following is the most likely main topic of the uploaded document?",
        "options": {
            "A": "A general overview of the subject",
            "B": "Unrelated random facts",
            "C": "A cooking recipe",
            "D": "A song lyric",
        },
        "answer": "A",
        "explanation": "Fallback question (AI service unavailable). The first option is generic-correct.",
    }


def build_prompt(context: str, asked: List[str]) -> str:
    asked_block = ""
    if asked:
        bullets = "\n".join(f"- {q}" for q in asked[-15:])
        asked_block = f"\nDo NOT repeat or paraphrase any of these previously asked questions:\n{bullets}\n"

    return f"""You are a quiz generator. Read the DOCUMENT below and create ONE multiple-choice question.

Rules:
- Question must be answerable using only the DOCUMENT.
- Provide exactly 4 options labeled A, B, C, D.
- Exactly one option must be correct.
- Keep the question concise.
- Output ONLY a single JSON object, no prose, no markdown fences.

JSON shape:
{{
  "question": "string",
  "options": {{"A":"string","B":"string","C":"string","D":"string"}},
  "answer": "A" | "B" | "C" | "D",
  "explanation": "short explanation grounded in the document"
}}
{asked_block}
DOCUMENT:
\"\"\"
{context}
\"\"\"
""".strip()


def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 600,
            "response_mime_type": "application/json",
        },
    )
    return (resp.text or "").strip()


# ---------- Routes ----------
@app.get("/")
def root():
    return {
        "name": "Quiz Backend",
        "version": "1.0.0",
        "endpoints": ["/health", "/upload", "/question", "/score"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai_configured": bool(GEMINI_API_KEY),
        "model": MODEL_NAME,
        "pdfs_loaded": len(PDF_STORE),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        data = await file.read()
        # Save temp file for PyPDF2
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
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        pdf_id = uuid.uuid4().hex
        PDF_STORE[pdf_id] = text[:200000]  # cap stored size
        return {"pdf_id": pdf_id, "pages": pages, "chars": len(text)}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/question")
def question(req: QuestionRequest):
    text = PDF_STORE.get(req.pdf_id)
    if not text:
        raise HTTPException(status_code=404, detail="pdf_id not found. Upload again.")

    context = text[:MAX_CONTEXT_CHARS]
    prompt = build_prompt(context, req.asked or [])

    if not GEMINI_API_KEY:
        return fallback_question()

    try:
        raw = call_gemini(prompt)
        parsed = safe_parse_json(raw)
        if not parsed or "question" not in parsed or "options" not in parsed or "answer" not in parsed:
            log.warning("Bad AI response, using fallback. Raw: %s", raw[:300])
            return fallback_question()

        # Normalize
        opts = parsed.get("options", {})
        if not all(k in opts for k in ("A", "B", "C", "D")):
            return fallback_question()
        ans = str(parsed.get("answer", "A")).strip().upper()
        if ans not in ("A", "B", "C", "D"):
            ans = "A"
        parsed["answer"] = ans
        parsed.setdefault("explanation", "")
        return parsed
    except Exception as e:
        log.exception("Gemini call failed")
        fb = fallback_question()
        fb["explanation"] = f"(AI error: {e}) " + fb["explanation"]
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
