# Team plan — 3-person split (A / B / C)

Team size: **3**. Deadline: **22 August 2026, 11:59 PM**. Submit **once**.

**Status (18 Aug 2026, final phase):** product pipeline + local UI are in. Extractive `latency_ms` is under 200ms locally (P100=169ms). Remaining is **ship**: Render live URL, two videos, `#RAGInGoa` posts, form.

This file is the **who-does-what**. The product map is [project_plan.md](project_plan.md). The brief is [project_requirement/task_2_hhg.md](../project_requirement/task_2_hhg.md).

---

## Stack freeze (locked)

- **No React, no Vue, no Vercel.**
- **One FastAPI app** serves HTML (`GET /`, `GET /sample-question`) and `POST /ask` + `POST /ask-text`.
- **STT:** Sarvam. **LLM:** Groq `openai/gpt-oss-20b`. **Embed:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Local URL:** `http://127.0.0.1:8000` (`uvicorn --host 0.0.0.0 --port 8000`).
- **Live URL (still to do):** one **Render Web Service**. Same origin. Index in `indexes/` loaded at start — do not download 55 GB on Render.

---

## What is done vs left

| Area | Status |
|---|---|
| A — Hindi download (`hinval` parquet, 5000 rows) | Done |
| A — Multi-strategy chunk + FAISS (`indexes/`, 289,561 chunks) | Done |
| B — FastAPI harness, Sarvam STT, retrieve, extractive + LLM, guards | Done (local) |
| B — Extractive path under 200ms (warmup, thread cap, lexical stopwords) | Done locally — P50=93 / P70=100 / P100=169 ms |
| B — P50/P70/P100 script (`app/latency_bench.py`) | Done — re-run 18 Aug on `http://127.0.0.1:8000` |
| C — English Tailwind mic UI, two answer cards, sample questions | Done (local `http://127.0.0.1:8000`) |
| C — Render live URL | **Open** |
| C — Videos + social + form | **Open** (22 Aug morning) |

Protect order if time is short: **working live demo → latency numbers in repo → videos → extra index polish**.

---

## Repo structure

```
root/
  one-time/download_dataset/       # A — Hindi (`hi`) download notes/scripts
  one-time/create_vector_database/ # A — chunk + embed + FAISS build
  indexes/                         # A output, B+Render consume: index.faiss, chunks.json
  app/                             # B — server, harness, STT, LLM, guardrails
  app/static/                      # C — HTML + vanilla JS (mic UI + /sample-question)
  team_plan.md
  project_plan.md
project_requirement/               # brief (repo root)
research_and_study/                # study notes (repo root)
```

No `backend/` or `front-end/` folders.

---

## How we split (rule)

Split by **feature ownership**, not by “Python vs React vs docs”.

| Person | Title | Owns |
|---|---|---|
| **A** | Data + index | Offline corpus → chunks → FAISS → `indexes/` |
| **B** | Pipeline + server | Live query path + Render start command |
| **C** | UI + ship | `app/static/` mic page, Render dashboard, videos, social, form |

If A is late, B cannot retrieve. If B is late, C cannot demo. API is frozen (see Shared). C’s remaining job is Render + videos + form.

---

## Shared (all three)

- One GitHub repo; branches + PRs; **no secrets in git** (`.env` local + Render env vars).
- Frozen API:

  `GET /` → HTML from `app/static/`  
  `GET /health`  
  `GET /sample-question` → sample questions page (from `chunks.json` selected Hindi `query_id`s)  
  `GET /api/sample-questions` → JSON list for that page and home chips  
  `POST /ask` — audio (wav/webm)  
  `POST /ask-text` — `{ "question": "..." }` (debug + bench)

  Response JSON (extractive-primary for 200ms):

  `{ "answer", "latency_ms", "path", "refused", "stt_ms", "retrieve_ms", "generate_ms", "extractive": {...}, "llm": {...}, "chunks": [...] }`

  `latency_ms` = retrieve + extractive + guards. **Never includes Groq.** `generate_ms` = Groq only. `stt_ms` = Sarvam.
- Every member posts **both** videos on Instagram, X, and LinkedIn with `#RAGInGoa`. At least one Instagram account public.
- Person **C** is submit owner (form + links). A and B freeze code **before** C records the demo.
- Render: bind `0.0.0.0` and use Render’s `PORT`. Wake the service before demo (free tier sleeps).

---

## Person A — Data + index

**Folders:** `one-time/download_dataset/`, `one-time/create_vector_database/`, output → `indexes/`

### In scope

1. **Download Hindi only** from [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) (`hi` config). Do not pull all 55.6 GB / all languages. Start `validation[:5000]`, then grow if time allows.
2. English comes **from the same Hindi rows** (`Eng_Query`, `Eng_Answer`, `English_passages`). No separate `en` download.
3. Extract **passage text** for the index (Hindi `Translated_passages` + English `English_passages`). Queries/answers are for testing, not the only indexed text.
4. **Chunking must be vast** (brief requirement): more than one naive fixed window. Include overlap, semantic vs fixed-size, metadata-aware (language, `query_id`, `is_selected`, source).
5. Embed with one model (e.g. `all-MiniLM-L6-v2`) — **same model B will use at query time**.
6. Build and save into `indexes/`:
   - `index.faiss` (vectors)
   - `chunks.json` (id → text + metadata)
   - `index_meta.json` (model name, dim, chunk count)
7. Document how to rebuild the index (README in `create_vector_database`).
8. Hand B a **frozen index path** and sample queries (`indexes/sample_queries.json`). Index must be small enough to ship on Render (current: 5k rows / ~290k chunks — watch Render RAM).

### Out of scope (A)

- FastAPI routes, Sarvam, LLM prompts, HTML/CSS, Render, videos.
- Rebuilding the index on every user query.

### Done when

- [x] Script produces `indexes/index.faiss` + `indexes/chunks.json`.
- [x] B can load those files without asking A how Hugging Face works.
- [x] README: Hindi parquet (not a Hub `hi` config), row count, strategies (`passage_full`, `fixed_overlap`, `sentence_group`), MiniLM, paths.

### Backup

B must know how to `faiss.read_index` and look up `chunks.json` if A is unavailable after freeze.

---

## Person B — Pipeline + server

**Folder:** `app/` (serves `app/static/` as `GET /`)

### In scope

1. FastAPI on `0.0.0.0` + `PORT` (local and Render).
2. Serve C’s static page from `/`. Implement `POST /ask`.
3. **STT:** Sarvam only (frozen). Audio in → transcript string. `stt_ms` separate.
4. **Input guardrails:** unsafe / empty → refuse **before** retrieve. Low-relevance retrieve → off-topic refuse; skip Groq.
5. **Retrieve:** same MiniLM as A, FAISS + lexical (skip stopwords) boost toward `passage_full`. Warm up encode+search at process start so the first UI click is not a PyTorch cold start (~2s). Cap torch/FAISS threads for single-query encodes.
6. **Generate:** **extractive** from chunks (200ms path) **and** Groq grounded rewrite (`LLM_FALLBACK=1`). Same chunks, separate output guards.
7. **Harness:** STT → guard → retrieve → extractive → LLM → output guard; retries on STT/LLM; structured JSON; Groq fail does not kill extractive.
8. **Output guardrails:** grounding overlap; exact refuse phrases (do not substring-match “not enough context” inside reasoning).
9. **Latency:** `app/latency_bench.py`. Report P50/P70/P100 on **`latency_ms` (extractive)**. Also log `generate_ms`. Target **under 200ms extractive**. Locked locally 18 Aug: P50=93 / P70=100 / P100=169. HTTP end-to-end waits on Groq so both cards return together — do not confuse that with `latency_ms`.
10. **Sample questions API:** `GET /api/sample-questions` only includes `query_id`s that have a selected Hindi `passage_full` chunk in `indexes/chunks.json`.
11. Render: `app/requirements.txt`, start command, env vars (`SARVAM_*`, `GROQ_*`, `LLM_FALLBACK`). Help C first successful deploy. **Still open.**

### Out of scope (B)

- Hugging Face download / chunk strategy design (A owns; B only **consumes** `indexes/`).
- Pixel-perfect HTML (C). Instagram posts, Google form.
- TTS (brief does not require speaking the answer back).
- React / Vercel.

### Done when

- [x] Text question → extractive + LLM (or refuse) without fancy UI (`/ask-text`).
- [x] Voice file → same path (`/ask`).
- [ ] Render URL shows C’s page and `/ask` works on that host.
- [x] Latency table exists (N queries). Local 18 Aug: extractive P50=93 / P70=100 / P100=169 ms. Re-run after Render and paste numbers in README.

### Backup

A understands retrieve enough to run B’s search function. C has the JSON example even if the server is down (mock).

---

## Person C — UI + ship

**Folder:** `app/static/`  
**Also owns:** Render Web Service (live URL), two videos, social checklist, [submission form](https://forms.gle/MNvCjcv23Hn2Eeu58)

### In scope

1. One HTML page + vanilla JS + Tailwind CDN (no React/Vue): mic, text debug, **two cards** (extractive vs LLM), latency chips (`retrieve` / `extractive` / `LLM`), sample-question chips, skeleton, toasts.
2. `GET /sample-question` lists grounded MSMARCO questions from `chunks.json`; click sends `/?q=...` into `/ask-text`.
3. `fetch('/ask')` with recorded audio; `fetch('/ask-text')` for typed questions. Loading and error states.
4. **Live working link on Render.** Test on a phone / second machine. Wake the free instance before recording the demo. **Open.**
5. **Video 1 (90s):** team **process**, not the product.
6. **Video 2:** end-to-end demo (speak → two answers) on the **Render URL**.
7. Chase every member: both videos on Instagram, X, LinkedIn, `#RAGInGoa`, one public IG.
8. Fill the form **once** when GitHub + Render URL + videos are final.

### Out of scope (C)

- FAISS build, chunking math, LLM prompt engineering, STT vendor integration (B).
- Separate Vercel/Netlify frontend.

### Done when

- [x] Local UI shows two grounded answers + timings (`http://127.0.0.1:8000`).
- [x] Sample questions page at `/sample-question`.
- [ ] Render URL works without a local Python install.
- [ ] Both videos exported; post checklist complete for all 3 members.
- [ ] Form submitted (C clicks submit only after A/B say freeze).

### Backup

If C is down: B can ship a minimal `app/static/index.html`. Videos still need a named owner — pick A as backup recorder.

---

## Remaining calendar

| When | A | B | C |
|---|---|---|---|
| **18 Aug (done)** | Index freeze v1 | Dual path + bench + extractive warmup (P100=169ms) | Local demo + `/sample-question` on port 8000 |
| **19–20 Aug** | RAM/size note for Render; no breaking index change unless B asks | `requirements.txt` + Render start; env docs; re-bench on host | **Render** Web Service; phone test |
| **21 Aug** | Hotfix index only if deploy OOM (smaller `--max-rows`) | Hotfix latency/keys | Live URL freeze; dry-run demo |
| **22 Aug morning** | Hotfix only | Hotfix only | Videos, posts, **one** form submit — not late night |

If something slips, protect in this order: **working live demo → latency numbers → videos → extra chunking polish**.

---

## Dependencies (do not invert)

```
A indexes/
  → B retrieve + generate + GET / + POST /ask
    → C static UI + Render live demo
      → videos + form
```

C can start on day 1 with mocks. C cannot **submit** until B’s app and A’s index are on Render.

---

## What we will not do

- Three people each building a full RAG clone.
- React + Vercel (or any second frontend host).
- Hugging Face Spaces as the default host (Gradio/Docker often needs a paid plan).
- Docs-only person with no demo owner.
- Changing FAISS format after 21 Aug without telling B and C.
- Resubmitting the form.

Names: fill in when the team assigns A / B / C.

| Role | Name |
|---|---|
| A | |
| B | |
| C | |
