# 🇮🇳 TruRank — Intelligent Candidate Discovery & Ranking

**An AI that ranks 100,000 candidates the way a great recruiter would — by understanding genuine *fit*, not matching keywords.**
TruRank is our entry to the Redrob **"India Runs" Track 1 — Intelligent Candidate Discovery & Ranking Challenge**. Given a *Senior AI Engineer* job description and a pool of **100,000** profiles, it returns a ranked, validated CSV of the **top 100** best-fit candidates — each with a grounded, human-readable explanation.

The scoring is dominated by the head of the list:
`Final = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`. Getting the top ~50 surgically right is the whole game.

---

## How We Beat Keyword Matching

The dataset is **adversarial by design** — it punishes the naive "embed the JD, embed each profile, sort by cosine" baseline. Four traps are planted:

| Trap | Signature | How TruRank beats it |
|---|---|---|
| **Keyword stuffer** | Non-tech title + many AI skills | Skill credit is *gated* behind career substance — skills only count if role descriptions corroborate them |
| **Plain-language hidden gem** | Genuine fit, zero buzzwords | 6 aspect-based semantic queries + BM25 over role *descriptions* surfaces them |
| **Behavioral twin** | Identical résumé, different availability | Bounded behavioral multiplier (0.65×–1.0×) separates them |
| **Honeypot (~80+)** | Internally-impossible profile | Consistency detector floors them to score = 0 |

---

## Architecture — Three-Stage Pipeline

```
       OFFLINE  (precompute.py — no time limit, downloads models)
 ┌──────────────────────────────────────────────────────────────────────┐
 │  candidates.jsonl  ──►  parse + embed (all-MiniLM-L6-v2)           │
 │                          └► candidates_cache.pkl                    │
 │  cross-encoder weights  ──► models/cross_encoder/                   │
 └──────────────────────────────────────────────────────────────────────┘
                                    │   (static artifacts)
                                    ▼
       ONLINE  (rank.py — ≤5 min, CPU-only, ZERO network, deterministic)
 ┌──────────────────────────────────────────────────────────────────────┐
 │  STAGE 1 — Retrieval                                                │
 │    ├─ 6 aspect-based JD queries (semantic, FAISS-free dot product)  │
 │    ├─ BM25 lexical retrieval over career descriptions               │
 │    └─ Reciprocal Rank Fusion (RRF, k=60) ──► 800 shortlist          │
 │                                                                     │
 │  STAGE 2 — Deep Scoring                                             │
 │    ├─ 13 structured signals (skill-gated, corroboration-checked)    │
 │    ├─ Behavioral multiplier (12 Redrob signals, bounded)            │
 │    ├─ Cross-encoder rerank (ms-marco-MiniLM-L-6-v2, top 200)       │
 │    └─ Honeypot detection ──► score floor                            │
 │                                                                     │
 │  STAGE 3 — Output                                                   │
 │    ├─ Top 100 (score desc, candidate_id tie-break)                  │
 │    ├─ Grounded per-candidate reasoning                              │
 │    └─ Write submission.csv                                          │
 └──────────────────────────────────────────────────────────────────────┘
```

### Stage 1 — Aspect-Based Retrieval with RRF

Instead of one coarse JD embedding, we decompose the JD into **6 independent aspect queries**:

| Aspect | What it retrieves |
|---|---|
| `retrieval_search` | embeddings, vector search, FAISS, Pinecone, Qdrant, Milvus |
| `ranking_eval` | NDCG, MRR, MAP, A/B testing, LTR, cross-encoder |
| `nlp_ir` | NLP, information retrieval, transformers, BERT, fine-tuning |
| `production_recency` | shipped, real users, scale, MLOps, serving |
| `product_company` | SaaS, startup, product ownership |
| `location_availability` | India, Pune, Noida, Bangalore, notice period |

Each aspect retrieves independently via semantic similarity. The 6 semantic rankings + 1 BM25 ranking + 1 aggregate semantic ranking are fused with **Reciprocal Rank Fusion (RRF, k=60)** across **8 independent rankings**. This scale-free method uses only rank positions, avoiding fragile normalization and surfacing hidden gems strong in specific aspects. This produces an **800-candidate shortlist**.

### Stage 2 — Cross-Encoder + 15-Signal Scoring

The shortlist is scored with **15 structured signals** (now including external validation and production recency). The top 200 candidates are then **reranked by a cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that reads each `(JD, candidate)` pair jointly with cross-attention — catching keyword stuffers that bi-encoder cosine similarity cannot.

**Signal breakdown (15 drivers, sweep-optimized on 200-candidate silver labels — NDCG@10 +8.3%):**
- `career_evidence` (0.200) — dominant signal, role substance from career text
- `location_fit` (0.100) — Pune/Noida/metro preference
- `semantic_similarity` (0.095) — weighted aspect-based semantic match
- `product_company_fit` (0.090) — product vs services background
- `skill_evidence` (0.082) — duration, proficiency, endorsements, assessments
- `experience_fit` (0.075) — 5-9 year band fit
- `domain_alignment` (0.070) — NLP/IR vs CV/speech domain fit
- `production_recency` (0.060) — ⭐ months since last shipping role
- `work_mode_fit` (0.050) — hybrid/onsite/remote preference
- `skill_recency` (0.046) — freshness of relevant skills
- `career_stability` (0.039) — sustained delivery vs title chasing
- `skill_corroboration` (0.032) — skills backed by career descriptions
- `external_validation` (0.029) — ⭐ GitHub, OSS, StackOverflow
- `culture_fit` (0.019) — culture signal alignment
- `skill_match` (0.014) — JD-required skill coverage

### Stage 3 — Honeypot Floor + Deterministic Output

Honeypot profiles (internally-impossible: 0-month experts, timeline contradictions) are detected and excluded. The top 100 non-honeypot candidates are written with grounded reasoning.

---

## The Numbers

Measured on the full **100,000**-candidate pool:

| Metric | Value |
|---|---|
| Total rank time | **~21s** (well under the 300s budget) |
| Compute | CPU-only, zero network calls at rank time |
| Honeypots caught | **80+** (0 in the top 100) |
| Shortlist (RRF-fused) | **800** candidates |
| Cross-encoder rerank | **200** pairs (the head, where NDCG lives) |
| CE score range | **0.087 – 0.993** (excellent discrimination) |
| Behavioral multiplier range | **0.50× – 1.15×** (23 Redrob signals) |
| Silver-label NDCG@10 | **0.952** (sweep-optimized, +8.3% vs hand-tuned) |
| Test suite | **67 tests passing** |
| Offline precompute | **~45 min** for 100K on CPU (one-time) |

---

## Reproduce

```bash
# 1. Environment (Python 3.11)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

```bash
# 2. OFFLINE precompute — run ONCE. Downloads the embedding model + cross-encoder
#    and builds data/processed/candidates_cache.pkl. No time limit.
python precompute.py --candidates data/raw/candidates.jsonl --out data/processed/candidates_cache.pkl
```

```bash
# 3. ONLINE ranking — the timed command (≤5 min, CPU, no network).
python rank.py --candidates data/raw/candidates.jsonl --cache data/processed/candidates_cache.pkl --out submission.csv
```

> Step 3 is the only step bound by the 5-minute / 16 GB / CPU-only budget.
> `rank.py` hard-locks the network off (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`)
> before any model import, so a stray fetch errors out rather than silently
> hitting the hub — a Stage-3 disqualifier.

---

## Live Demo

A beautifully themed, interactive dashboard to visualize the top candidates, their semantic match, signal breakdown, honeypot detection, and generated reasoning.

🔗 **Streamlit Cloud Sandbox:** [https://trurank-indiaruns.streamlit.app/](https://trurank-indiaruns.streamlit.app/)

To run it locally:
```bash
streamlit run app/demo.py
```

---

## Repository Layout

```text
india_runs_track1/
├── README.md
├── requirements.txt
├── config.yaml                # All weights, thresholds, JD alignment
├── submission_metadata.yaml   # Challenge submission metadata
├── data/
│   ├── raw/                   # Original dataset files
│   └── processed/             # Precomputed embeddings cache
├── models/
│   └── cross_encoder/         # Cached cross-encoder weights (gitignored)
├── src/
│   ├── jd_parser.py           # JD understanding + feature extraction
│   ├── profile_parser.py      # Candidate profile normalization
│   ├── embeddings.py          # Semantic embedding pipeline
│   ├── signals.py             # 13-signal scoring (incl. skill corroboration)
│   ├── cross_encoder.py       # Cross-encoder reranker (CPU, offline-safe)
│   ├── honeypot_detector.py   # Trap detection (deterministic, REFERENCE_DATE)
│   ├── explainer.py           # Natural language explanations per candidate
│   └── utils.py               # Logging, config, helpers
├── app/
│   └── demo.py                # Streamlit UI dashboard
├── precompute.py              # Step 1: Embedding + cross-encoder download
├── rank.py                    # Step 2: RRF retrieval + CE rerank + ranking
└── submission.csv             # Final generated output
```

---

## Tech Stack

- **Python 3.11**, fully deterministic (static `REFERENCE_DATE = 2026-06-22` for all date math, stable sorts, `candidate_id` tie-break).
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), 384-dim.
- **Lexical retrieval:** `rank-bm25` (BM25Okapi over career descriptions).
- **Rank fusion:** Reciprocal Rank Fusion (RRF, k=60) — Cormack et al., 2009.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU, top 200 pairs).
- **Network isolation:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` hard-locked before any model import.

---

## AI Tools Used

- **Google Gemini** — primary architecture design, code generation, signal engineering, and iterative debugging.
- **Anthropic Claude** — code review, determinism audit, cross-encoder integration guidance, and README refinement.

---

## License

MIT
