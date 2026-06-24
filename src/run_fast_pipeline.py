import os
import json
import time
import numpy as np
import faiss
import polars as pl
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.hard_filter import apply_hard_filter
from src.score_career import compute_A, compute_keyword_max
from src.score_skills import compute_B
from src.score_embed import compute_C_all
from src.availability import apply_multipliers
from src.output import write_submission, generate_reasoning
from src.precompute import build_candidate_text, build_jd_text

def run_fast_pipeline():
    start_time = time.perf_counter()
    
    print("=" * 60)
    print("RUNNING HIGH-PERFORMANCE PIPELINE ON 100,000 CANDIDATES")
    print("=" * 60)
    
    # Paths
    jd_path = Path("data/job_description.json")
    candidates_path = Path("[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    output_path = Path("submission.csv")
    
    if not jd_path.exists():
        raise FileNotFoundError(f"Job description not found at {jd_path}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates dataset not found at {candidates_path}")
        
    # 1. Load Job Description and candidates
    print("Loading Job Description...")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd = json.load(f)
        
    print("Loading 100,000 candidates (streamed from JSONL)...")
    candidates = []
    titles = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            candidates.append(cand)
            title = cand.get("profile", {}).get("current_title") or cand.get("current_title") or ""
            titles.append(title)
            
    print(f"Loaded {len(candidates)} candidates.")
    
    # 2. Fit TF-IDF on all candidate titles for the hard filter
    print("Fitting TF-IDF Vectorizer on all candidate titles...")
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    tfidf.fit(titles)
    
    # 3. Apply Hard Filters (Stage 1)
    print("\n--- STAGE 1: Applying Hard Filters ---")
    survivors, killed = apply_hard_filter(candidates, jd, tfidf)
    print(f"Killed: {len(killed)} candidates.")
    print(f"Surviving: {len(survivors)} candidates.")
    
    if not survivors:
        print("No candidates survived. Exiting.")
        return
        
    # 4. Generate Embeddings for ONLY the survivors + JD (Massive speedup!)
    print("\n--- STAGE 2: Embedding Survivors & Job Description ---")
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Embedding Job Description...")
    jd_text = build_jd_text(jd)
    jd_vec = model.encode(jd_text, normalize_embeddings=True).astype("float32")
    
    print(f"Embedding {len(survivors)} surviving candidates...")
    survivor_texts = [build_candidate_text(s) for s in survivors]
    cand_vecs = model.encode(survivor_texts, batch_size=256, normalize_embeddings=True).astype("float32")
    
    # 5. Compute Stage 2C Embedding Similarity
    print("Computing embedding similarities (Stage 2C)...")
    C_scores = compute_C_all(jd_vec, cand_vecs)
    C_map = {str(s.get("candidate_id") or s.get("id")): float(score) for s, score in zip(survivors, C_scores)}
    
    # 6. Fit TF-IDF on survivor texts for career scoring
    print("Fitting TF-IDF Vectorizer on survivor texts...")
    tfidf_surv = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    tfidf_surv.fit(survivor_texts)
    
    # 7. Compute Stage 2A & 2B scores
    print("Computing career and skill scores...")
    keyword_max = compute_keyword_max(survivors, jd, tfidf_surv)
    
    raw_scored = []
    for cand in survivors:
        cand_id = str(cand.get("candidate_id") or cand.get("id"))
        A_res = compute_A(cand, jd, tfidf_surv, keyword_max)
        B_res = compute_B(cand, jd)
        C = C_map.get(cand_id, 0.0)
        
        A = A_res["A"]
        B = B_res["B"]
        raw_score = round(0.40 * A + 0.35 * B + 0.25 * C, 4)
        
        raw_scored.append({
            "candidate_id": cand_id,
            "candidate": cand,
            "A": A,
            "title_sim": A_res["title_sim"],
            "industry_match": A_res["industry_match"],
            "prod_keyword_density": A_res["prod_keyword_density"],
            "yoe_score": A_res["yoe_score"],
            "B": B,
            "must_have_coverage": B_res["must_have_coverage"],
            "nice_coverage": B_res["nice_coverage"],
            "cert_bonus": B_res["cert_bonus"],
            "C": C,
            "raw_score": raw_score
        })
        
    # 8. Apply Behavioral Multipliers (Stage 3)
    print("\n--- STAGE 3: Applying Behavioral Multipliers ---")
    final_scored = apply_multipliers(raw_scored, jd)
    
    # 9. Output ranked results (Stage 4)
    print("\n--- STAGE 4: Generating Output ---")
    df_final = write_submission(final_scored, jd, str(output_path))
    
    elapsed_time = time.perf_counter() - start_time
    print("=" * 60)
    print(f"PIPELINE EXECUTED IN {elapsed_time:.2f} SECONDS")
    print("=" * 60)
    
    print("\nTOP 10 CANDIDATES:")
    print(df_final.head(10))

if __name__ == "__main__":
    run_fast_pipeline()
