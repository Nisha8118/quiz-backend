# Quiz App v4 — Adaptive AI Quiz

Two modes:
- **By Subject** — pick any subject + difficulty, AI generates questions on the fly.
- **From PDF** — upload a PDF, AI generates questions only from its content.

## What changed in v4 (fixes your bugs)

1. **404 model error fixed** — `gemini-1.5-flash` was retired on Google's free `v1beta` endpoint. Backend now auto-probes a list of current models (`gemini-flash-latest`, `gemini-2.5-flash`, `gemini-2.0-flash`) and uses the first one that works.
2. **Subject mode added** — students can now take a quiz on any subject without uploading a PDF. PDF is optional.
3. **No repeated questions** — backend tracks `asked` list and instructs the AI to avoid them. Fallback bank now has 8 varied questions instead of 1.
4. **Quiz length picker** — choose 5/10/15/20 questions, see progress bar and final score.

---

## Backend — deploy to Render

1. Push the `backend/` folder to a GitHub repo (`quiz-backend`).
2. Render → New → Web Service → connect repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Environment variable: `GEMINI_API_KEY = your AIza... key`
6. Instance: Free.
7. After deploy, open `https://your-app.onrender.com/health` — you should see `"ai_configured": true` and an `active_model` once a question is requested.
8. To list every model your key can use: open `/models`.

If something fails, check `/health` and `/models`.

## Frontend — deploy to Vercel

1. Edit `frontend/script.js` → set `const API_URL = "https://your-app.onrender.com";`
2. Push the `frontend/` folder to a GitHub repo (`quiz-frontend`).
3. Vercel → Import repo → Deploy (framework preset: **Other**).
4. Done.

---

## Local test

```bash
cd backend
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
uvicorn app:app --reload
# open frontend/index.html in a browser (set API_URL to http://localhost:8000)
```
