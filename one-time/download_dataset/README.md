# Download MSMARCO-XI (Hindi only)

This folder is for **one-time** download of the RAG source data.

Dataset: [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)

Do **not** download the full 55.6 GB dump. That total includes every Indic language. For this project we load **only the `hi` (Hindi) config**.

There is no separate `en` config. Each Hindi row already includes the original English fields (`Eng_Query`, `Eng_Answer`, `English_passages`).

---

## 1. Install

From this folder:

```bash
pip install -r requirements.txt
```

(`datasets` + `pyarrow`)

---

## 1b. Run the downloader

Default = Hindi `hi`, `validation[:5000]`:

```bash
python download.py
```

Same as:

```bash
python download.py --config hi --split validation --limit 5000
```

Other useful calls:

```bash
python download.py --limit 5
python download.py --split validation --limit 0
python download.py --split train --limit 5000
```

The Hub builder only has config `default`. Hindi is **`validation/hinval.parquet`** / **`train/hintrain.parquet`**, not `load_dataset(..., "hi")`. `download.py` maps `--config hi` to those files.

Output JSONL (gitignored):

`data/msmarco_xi_hi_{split}_{n}.jsonl`

Example: `data/msmarco_xi_hi_validation_5000.jsonl`

Each line has `query_id`, Hindi `query` / `Answer` / `Translated_passages`, English `Eng_Query` / `Eng_Answer` / `English_passages`, and `is_selected`. The next step (`../create_vector_database`) reads this file.

---

## 2. Load Hindi only

```python
from datasets import load_dataset

# Hindi config only — not the full multilingual corpus
dataset = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")
```

First run downloads files from Hugging Face and caches them on disk (usually under `~/.cache/huggingface/datasets/` on Linux/macOS, or `%USERPROFILE%\.cache\huggingface\datasets\` on Windows). Later runs use the cache.

---

## 3. Start small (recommended)

Do not pull the entire Hindi `train` split on the first try.

```python
from datasets import load_dataset

# First 5,000 validation rows — enough to inspect fields and build a trial index
ds = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="validation[:5000]")
```

Useful splits:

| Split | Use |
|---|---|
| `validation[:5000]` | Inspect + first FAISS trial |
| `validation` | Full Hindi validation |
| `train` | Large; use only after a small slice works |

---

## 4. What you get in one Hindi row

Hindi:

- `query`
- `Answer`
- `passages["Translated_passages"]`

English (same row):

- `Eng_Query`
- `Eng_Answer`
- `passages["English_passages"]`

Also: `query_id`, `query_type`, `passages["is_selected"]`, translation `meta`.

For the vector index, use **passage text**, not only the short `Answer`.

```python
row = ds[0]

print(row["query"])
print(row["Eng_Query"])
print(row["passages"]["Translated_passages"][0])
print(row["passages"]["English_passages"][0])
```

---

## 5. What not to do

- Do not load every language (`bn`, `ta`, …) unless you explicitly need them.
- Do not stream the full dataset at query time. Download once here, then build FAISS in `../create_vector_database`.
- Do not assume Hugging Face already stored vectors. This dataset is **text**. Embeddings are created later.

---

## 6. Offline after cache

Once `download.py` has finished:

- Hugging Face still caches parquet under `~/.cache/huggingface/datasets/` (or `%USERPROFILE%\.cache\huggingface\datasets\` on Windows).
- **This folder’s JSONL** (`data/msmarco_xi_hi_*.jsonl`) is what `create_vector_database` should read.
- The live app should load `indexes/` (`index.faiss` + chunk metadata), not Hugging Face, on every user question.
