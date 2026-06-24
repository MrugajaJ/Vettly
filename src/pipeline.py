import os
import json
import pickle
import time
from pathlib import Path

from src.hard_filter import apply_hard_filter
from src.raw_score import compute_raw_scores
from src.availability import apply_multipliers
from src.output import write_submission

def run_pipeline(candidates_path: str, jd_path: str, output_path: str):
    """Executes the complete candidate vetting, scoring, and ranking pipeline."""
    start_time = time.perf_counter()
    
    print("=" * 60)
    print("STARTING VETTLY CANDIDATE SCORING AND FILTERING PIPELINE")
    print("=" * 60)
    
    # Resolve paths
    cand_path = Path(candidates_path)
    job_desc_path = Path(jd_path)
    out_path = Path(output_path)
    
    # Fallback to vettly/ if not found directly (handles execution from workspace root vs. vettly root)
    if not cand_path.exists() and Path("vettly").joinpath(cand_path).exists():
        cand_path = Path("vettly").joinpath(cand_path)
    if not job_desc_path.exists() and Path("vettly").joinpath(job_desc_path).exists():
        job_desc_path = Path("vettly").joinpath(job_desc_path)
        
    print(f"Candidates Path: {cand_path.resolve()}")
    print(f"Job Description Path: {job_desc_path.resolve()}")
    print(f"Output Path: {out_path.resolve()}")
    
    # 1. Load candidates and job description
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidates file not found at: {cand_path}")
    if not job_desc_path.exists():
        raise FileNotFoundError(f"Job description file not found at: {job_desc_path}")
        
    print("Loading datasets...")
    with open(cand_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    with open(job_desc_path, "r", encoding="utf-8") as f:
        jd = json.load(f)
        
    print(f"Loaded {len(candidates)} candidates.")
    
    # 2. Load TF-IDF vectorizer
    # TF-IDF resides in data/precomputed/tfidf.pkl relative to candidates directory
    precomputed_dir = cand_path.parent / "precomputed"
    tfidf_path = precomputed_dir / "tfidf.pkl"
    
    if not tfidf_path.exists():
        # Try alternate location
        script_dir = Path(__file__).resolve().parent
        tfidf_path = script_dir.parent / "data" / "precomputed" / "tfidf.pkl"
        
    print(f"Loading TF-IDF Vectorizer from: {tfidf_path}")
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
        
    # 3. Run Hard Filter (Stage 1)
    print("\n--- STAGE 1: Applying Hard Filters ---")
    survivors, killed = apply_hard_filter(candidates, jd, tfidf)
    print(f"Killed: {len(killed)} candidates.")
    print(f"Surviving: {len(survivors)} candidates.")
    
    if not survivors:
        print("WARNING: No candidates survived the hard filters! Pipeline exiting early.")
        return
        
    # 4. Run Raw Scoring (Stage 2A, 2B, 2C in parallel)
    print("\n--- STAGE 2: Computing Raw Scores ---")
    raw_scored = compute_raw_scores(survivors, jd)
    
    # 5. Run Behavioral Multipliers (Stage 3)
    print("\n--- STAGE 3: Applying Behavioral Multipliers ---")
    final_scored = apply_multipliers(raw_scored, jd)
    
    # 6. Run Output Generation (Stage 4)
    print("\n--- STAGE 4: Generating Output ---")
    df_final = write_submission(final_scored, jd, str(out_path))
    
    elapsed_time = time.perf_counter() - start_time
    print("=" * 60)
    print(f"PIPELINE EXECUTED IN {elapsed_time:.2f} SECONDS")
    print("=" * 60)
    
    # Print the top 10 rows from the output DataFrame
    print("\nTOP 10 CANDIDATES:")
    print(df_final.head(10))

if __name__ == "__main__":
    # Default paths assuming execution from workspace root
    default_candidates = "data/candidates.json"
    default_jd = "data/job_description.json"
    default_output = "submission.csv"
    
    run_pipeline(default_candidates, default_jd, default_output)
