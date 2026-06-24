import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.score_career import compute_A, compute_keyword_max
from src.score_skills import compute_B
from src.score_embed import load_artifacts, get_C_map

def candidate_worker(candidate: dict, jd: dict, tfidf, keyword_max: float) -> tuple[dict, dict, dict]:
    """Module-level worker function to score a single candidate (picklable for ProcessPoolExecutor)."""
    A_res = compute_A(candidate, jd, tfidf, keyword_max)
    B_res = compute_B(candidate, jd)
    return candidate, A_res, B_res

def compute_raw_scores(survivors: list[dict], jd: dict) -> list[dict]:
    """Hub module that coordinates Stage 2A, 2B, and 2C to compute raw scores in parallel."""
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vettly_dir = os.path.dirname(script_dir)
    precomputed_dir = os.path.join(vettly_dir, "data", "precomputed")
    
    tfidf_pkl_path = os.path.join(precomputed_dir, "tfidf.pkl")
    
    print("Loading TF-IDF vectorizer...")
    with open(tfidf_pkl_path, "rb") as f:
        tfidf = pickle.load(f)
        
    print("Loading embedding artifacts...")
    jd_vec, cand_vecs, cand_ids = load_artifacts(precomputed_dir)
    
    print("Computing embedding similarities (Stage 2C)...")
    C_map = get_C_map(jd_vec, cand_vecs, cand_ids)
    
    print("Computing pool-wide keyword density maximum...")
    keyword_max = compute_keyword_max(survivors, jd, tfidf)
    
    results = []
    
    # Process in parallel
    print(f"Parallelizing scoring for {len(survivors)} candidates across CPU cores...")
    max_workers = os.cpu_count() or 4
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(candidate_worker, cand, jd, tfidf, keyword_max): cand
            for cand in survivors
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring candidates"):
            try:
                candidate, A_res, B_res = future.result()
                cand_id = str(candidate.get("id") or candidate.get("candidate_id") or "")
                
                # Get Stage 2C embedding score
                C = C_map.get(cand_id, 0.0)
                
                # Assemble raw score: 0.40*A + 0.35*B + 0.25*C
                A = A_res["A"]
                B = B_res["B"]
                raw_score = round(0.40 * A + 0.35 * B + 0.25 * C, 4)
                
                results.append({
                    "candidate_id": cand_id,
                    "candidate": candidate,
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
            except Exception as e:
                print(f"Error scoring candidate: {e}")
                
    return results
