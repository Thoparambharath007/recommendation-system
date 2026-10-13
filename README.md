# Metadata & Review-Based Hybrid Apparel Recommendation System

A natural-language product search & recommendation engine over a **vision-verified
catalog of 50,224 men's shirts**, built on the methodology of:

> Roy, Kumar & Kumar — *"Metadata and Review-Based Hybrid Apparel Recommendation System Using Cascaded Large Language Models"* (IEEE Access, Vol. 12, 2024)

Dataset: **2023 UCSD Amazon Review Dataset** (7.2M metadata records → 63,190 rule-segregated
shirt candidates → 50,224 CLIP-verified shirts).

---

## Current architecture (the query layer — since Oct 2026)

```
plain-English question
   -> LLM generates SQL        (NVIDIA Nemotron 3.5 Lightning, OpenAI-compatible API)
   -> AST validation           (SQLGlot: SELECT parent_asin FROM shirts only —
                                joins / subqueries / mutations / unknown columns rejected)
   -> DuckDB scans Parquet     (shirts_master.parquet, serverless, no DB)
   -> ranked product cards     (price-provenance tier -> Bayesian rating -> review count)
```

Everything lives in [`DuckDB idea/`](DuckDB%20idea/README.md):

| Path | What it is |
|---|---|
| `DuckDB idea/Start_Query_App.bat` | **THE ONE-CLICK ENTRY POINT** — double-click to run |
| `DuckDB idea/app.py` + `web/` | FastAPI server + single-page search UI (http://127.0.0.1:8010) |
| `DuckDB idea/llm_sql.py`, `request_policy.py` | LLM -> SQL adapter with provider-safe JSON handling |
| `DuckDB idea/query_engine.py` | SQL validation, exact colour-token matching, ranking |
| `DuckDB idea/data/shirts_master.parquet` | The 50,224-shirt master catalog (tracked in git) |
| `DuckDB idea/01…08_*.py`, `merge_vision.py` | Scripts that built the catalog (CLIP verification, price estimation, review mining, Kaggle v3.2 colour/pattern merge) |
| `DuckDB idea/QUERY_APP.md` | Config, restrictions, ranking and test details |
| `SRS.md` / `RESEARCH.md` | Software requirements spec & research notes |

## Run it (clone → recommendations)

1. Install Python **3.11+** (tick "Add to PATH").
2. `git clone https://github.com/Thoparambharath007/recommendation-system.git`
3. `cd recommendation-system/"DuckDB idea"` and double-click **`Start_Query_App.bat`**.
4. Copy `.env.example` to `.env`, add your own free NVIDIA API key
   (`nvidia/nemotron-3.5-lightning-30b-a3b`; OpenRouter also supported — see QUERY_APP.md).
5. Open **http://127.0.0.1:8010** and ask, e.g. *"plain maroon casual shirt under 1500"*.

The catalog ships inside the repo — no dataset download needed for the query app.
(Intermediate build data — CSV copies, pre-merge backups, review mining outputs —
stays off git by design; ask the team for the OneDrive copy if you need to rebuild.)

## Tests

```bash
python -m unittest discover -s "DuckDB idea/tests" -v   # 16 tests
```

Covers catalog-merge identity, duplicate/missing IDs, exact shade matching, unsafe-SQL
rejection, real-Parquet ranking, the API adapter and FastAPI behaviour. Live provider
checks are recorded in `DuckDB idea/docs/live-provider-check.json`.

## Honest scope

This is a working query application with a transparent deterministic ranking baseline —
not yet an evaluated recommender. Evaluation metrics (Recall@10, NDCG@10, MRR), the
verified-purchase review join and sentiment ranking are the next milestones.
Button colour, live stock, delivery and exact sizes are absent from the source data
and explicitly unsupported rather than guessed.

The previous 7-phase cascaded-LLM pipeline (src/, web/, scripts/ of the old design)
was retired in Oct 2026; its full code and history remain in earlier commits.

## Team

- T BharathChandra
- S Jagruth
- S Vijaya
- K PavanBalu
- N YagnaSai

Mentor: **T Venu**
