# Create vector database (offline)

This folder turns the JSONL from [`../download_dataset`](../download_dataset/README.md) into a FAISS index.

It runs **once** (or whenever you rebuild). The live app must **not** re-download Hugging Face or re-embed the corpus on every question.

Output (consumed by Person B / Render):

| File | What it is |
|---|---|
| [`../../indexes/index.faiss`](../../indexes/index.faiss) | Vectors + nearest-neighbour index |
| [`../../indexes/chunks.json`](../../indexes/chunks.json) | `id` → chunk text + metadata |
| [`../../indexes/index_meta.json`](../../indexes/index_meta.json) | Embedding model name, dim, strategies |
| [`../../indexes/sample_queries.json`](../../indexes/sample_queries.json) | Example Hindi/English questions for retrieval tests |

---

## 1. Prerequisite

```bash
cd ../download_dataset
pip install -r requirements.txt
python download.py
```

You need a file like `../download_dataset/data/msmarco_xi_hi_validation_5000.jsonl`.

---

## 2. Install

From this folder:

```bash
pip install -r requirements.txt
```

(`sentence-transformers`, `faiss-cpu`, `numpy`)

First `build_index.py` run also downloads `all-MiniLM-L6-v2` into the Hugging Face cache. CPU is enough. Do not use Ollama here.

---

## 3. Build

```bash
python build_index.py
```

Uses the newest `msmarco_xi_hi_*.jsonl` under `download_dataset/data/`.

Trial (faster):

```bash
python build_index.py --max-rows 200
```

Explicit input / output:

```bash
python build_index.py --input ../download_dataset/data/msmarco_xi_hi_validation_5000.jsonl --out-dir ../../indexes
```

---

## 4. Chunking (not one naive window)

Each Hindi and English **passage** is split with **three** strategies, then de-duplicated:

| Strategy | What |
|---|---|
| `passage_full` | Whole MSMARCO passage (natural document unit) |
| `fixed_overlap` | 400-character windows, 80-character overlap (only if the passage is longer than 400) |
| `sentence_group` | Consecutive sentences grouped (2 at a time) — not a single fixed char window |

Every chunk stores metadata for later retrieval / guardrails:

- `language` (`hi` / `en`)
- `query_id`, `query_type`
- `passage_index`, `is_selected` (MSMARCO relevance flag)
- `strategy`

Queries and short `Answer` fields are **not** the main index text (they are in `sample_queries.json` for tests). Passages are what we embed.

---

## 5. Vectors

- Model: `sentence-transformers/all-MiniLM-L6-v2` (same one Person B must load at query time)
- Vectors are **L2-normalized**
- FAISS type: `IndexFlatIP` (inner product = cosine after normalize)

Query time:

1. `model.encode(question, normalize_embeddings=True)`
2. `index.search(query_vector, k)`
3. Use returned integer ids to look up `chunks.json`

---

## 6. What not to do

- Do not rebuild this index inside `POST /ask`.
- Do not ship the Hugging Face 55 GB cache to Render — only `indexes/`.
- Do not change `EMBED_MODEL` in `build_index.py` without changing the app encoder too.

Large `index.faiss` / `chunks.json` are gitignored. Rebuild on the machine that deploys, or copy those two files onto Render.
