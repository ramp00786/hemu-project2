# App (Person B + C)

One Python app serves both:

- `GET /` -> `app/static/index.html`
- `POST /ask` -> audio path (Sarvam STT -> retrieve -> extractive + LLM)
- `POST /ask-text` -> text debug/bench path

No React/Vercel. Same-origin UI + API.

## Query path (requirement-aligned)

After a transcript exists:

1. Input guard
2. FAISS + lexical retrieve
3. **Extractive answer** from retrieved passages (no remote LLM)
4. Output / grounding guard on extractive
5. **Explicit Groq LLM card** on the same chunks (`LLM_FALLBACK=1`), with its own output guard

The UI shows both answers side by side.

`latency_ms` is retrieve + extractive + guards only. It never includes Groq. `generate_ms` is Groq only. `stt_ms` is separate because Sarvam STT is required by the brief and is a network call.

The 200ms target is the extractive path (`path: "extractive"`, field `latency_ms`). HTTP end-to-end time is slower because the request waits for Groq so both cards can return together.

## Environment

Create `root/.env` (do not commit):

```env
SARVAM_API_KEY=...
SARVAM_STT_URL=...
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
LLM_FALLBACK=1
```

`LLM_FALLBACK=1` enables the LLM card. `0` skips Groq; extractive still works.

## Install

From `root` venv:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r app\requirements.txt
```

## Run local

```powershell
cd root
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open: `http://127.0.0.1:8000`

Health:

- `GET /health`

## Latency bench

With server running:

```powershell
python app\latency_bench.py --base-url http://127.0.0.1:8000 --max-queries 30
```

Prints P50/P70/P100 for HTTP time, retrieve_ms, extractive `latency_ms` (200ms target), and `generate_ms` (LLM, not the 200ms claim).

## Important

- Query encoder must match index model: `sentence-transformers/all-MiniLM-L6-v2`
- App reads prebuilt files from `root/indexes/`
- Do not rebuild FAISS at request time
