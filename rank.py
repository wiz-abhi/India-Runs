import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import csv
import pickle
import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.jd_parser import JobDescription
from src.profile_parser import CandidateProfile
from src.signals import SignalComputer
from src.explainer import ExplainerEngine
from src.honeypot_detector import HoneypotDetector
from src.utils import load_config


def run_ranking(cache_path: str, output_path: str):
    start_time = time.time()
    print("Loading configuration...")
    cfg = load_config()
    target_jd_cfg = cfg.get("target_jd", {})

    # Create the JD representation
    jd = JobDescription(
        raw_text="Target Hackathon JD",
        required_skills=target_jd_cfg.get("required_skills", []),
        preferred_skills=target_jd_cfg.get("preferred_skills", []),
        min_experience_years=target_jd_cfg.get("min_experience", 4),
        domain=target_jd_cfg.get("domain", "data_science"),
        culture_signals=target_jd_cfg.get("culture_signals", []),
    )
    jd_embedding_text = """
Senior AI/ML Engineer — Production Retrieval, Ranking and Recommendation Systems.
Seeking engineers who have shipped embedding-based retrieval, semantic search, or
recommendation systems to real users at scale. Must demonstrate: production deployment
of dense retrieval or hybrid search (Pinecone, Weaviate, Qdrant, Milvus, FAISS,
Elasticsearch); evaluation with NDCG, MRR, MAP, A/B testing, offline-to-online
correlation; handling embedding drift, index refresh, retrieval quality regression.
Strong Python. Product company background preferred. 5-9 years applied ML experience.
Location: Pune or Noida preferred. Immediate to 30-day notice. NLP/IR domain expertise.
LTR, fine-tuning (LoRA/QLoRA/PEFT), or distributed systems a strong plus.
"""

    print(f"Loading precomputed cache from {cache_path}...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    profiles: List[CandidateProfile] = cache_data["profiles"]
    embeddings: np.ndarray = cache_data["embeddings"]
    model_name = cache_data["model"]

    print(
        f"Loaded {len(profiles)} candidates. Loading model '{model_name}' for JD embedding..."
    )
    model = SentenceTransformer(model_name)
    jd_embedding = model.encode(jd_embedding_text, normalize_embeddings=True)

    print("Computing semantic similarities (vectorized)...")
    # embeddings and jd_embedding are both normalized
    jd_norm = np.asarray(jd_embedding, dtype=np.float32)
    prof_norms = np.asarray(embeddings, dtype=np.float32)
    semantic_scores = prof_norms @ jd_norm

    from rank_bm25 import BM25Okapi

    # BM25 query: key terms from the JD
    BM25_QUERY_TERMS = [
        "retrieval", "ranking", "embedding", "semantic", "search", "recommendation",
        "vector", "production", "deployed", "ndcg", "mrr", "evaluation", "faiss",
        "pinecone", "qdrant", "weaviate", "elasticsearch", "pipeline", "scale"
    ]

    def get_description_text(profile):
        return " ".join(profile.career_history_desc).lower()

    print("Building BM25 index over career descriptions...")
    corpus = [get_description_text(p).split() for p in profiles]
    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(BM25_QUERY_TERMS)

    # TWO-STAGE RETRIEVAL: Combine semantic similarity + lexical matching
    TOP_K = 2500
    print(f"Filtering top {TOP_K} semantic and {TOP_K} lexical candidates...")
    
    # argpartition is faster than argsort for just getting top K
    semantic_top = set(np.argpartition(-semantic_scores, min(TOP_K, len(profiles)) - 1)[:min(TOP_K, len(profiles))])
    bm25_top = set(np.argpartition(-bm25_scores, min(TOP_K, len(profiles)) - 1)[:min(TOP_K, len(profiles))])
    top_k_indices = list(semantic_top | bm25_top)
    print(f"Union shortlist size: {len(top_k_indices)} candidates")
    
    print("Computing behavioral signals and final composites for shortlist...")
    computer = SignalComputer()
    scored_candidates = []

    for i in top_k_indices:
        profile = profiles[i]
        sim = float(semantic_scores[i])
        scores = computer.compute_all(profile, jd, sim)
        # Round to 4 decimal places so the sort order matches what validate_submission.py sees
        scored_candidates.append((scores.composite_score, profile.candidate_id, profile, scores))

    # Sort: Primary by score (descending), Secondary by candidate_id (ascending)
    print("Sorting candidates...")
    scored_candidates.sort(key=lambda x: (-x[0], x[1]))

    print("Running honeypot detection on top candidates...")
    detector = HoneypotDetector()
    top_100 = []
    honeypots_caught = 0
    
    # Keep pulling until we have 100 safe candidates
    for score, cid, profile, scores in scored_candidates:
        if len(top_100) >= 100:
            break
            
        # Detect honeypots
        hp_result = detector.detect(profile.raw)
        if hp_result.is_honeypot:
            honeypots_caught += 1
            continue
            
        top_100.append((score, cid, profile, scores))
        
    print(f"Caught and filtered {honeypots_caught} honeypots from the top ranking.")

    print("Generating explanations for top 100...")
    explainer = ExplainerEngine()
    results = []

    for rank, (score, cid, profile, scores) in enumerate(top_100, start=1):
        # Generate reasoning based on the new SignalScores structure
        reasoning = explainer.explain_rank(profile, scores, jd, rank=rank, total=len(top_100))
        # Format the reasoning properly to fit in CSV (no newlines in the cell)
        reasoning = reasoning.replace("\n", " ").replace("\r", "").strip()
        results.append(
            {
                "candidate_id": cid,
                "rank": rank,
                "score": f"{score:.6f}",
                "reasoning": reasoning,
            }
        )

    print(f"Writing output to {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Done in {time.time() - start_time:.2f}s total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank candidates from cache")
    parser.add_argument(
        "--candidates",
        type=str,
        default="data/raw/candidates.jsonl",
        help="Input candidates file (ignored in cached mode)",
    )
    parser.add_argument(
        "--out", type=str, default="submission.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--cache", type=str, default="data/processed/candidates_cache.pkl", help="Input cache file"
    )
    args = parser.parse_args()

    run_ranking(args.cache, args.out)
