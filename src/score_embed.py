import os
import json
import numpy as np

def load_artifacts(precomputed_dir: str) -> tuple[np.ndarray, np.ndarray, list]:
    """Loads the precomputed artifacts: jd_vec.npy, cand_vecs.npy, and cand_ids.json."""
    jd_vec_path = os.path.join(precomputed_dir, "jd_vec.npy")
    cand_vecs_path = os.path.join(precomputed_dir, "cand_vecs.npy")
    cand_ids_path = os.path.join(precomputed_dir, "cand_ids.json")
    
    jd_vec = np.load(jd_vec_path)
    cand_vecs = np.load(cand_vecs_path)
    
    with open(cand_ids_path, "r", encoding="utf-8") as f:
        cand_ids = json.load(f)
        
    return jd_vec, cand_vecs, cand_ids

def compute_C_all(jd_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
    """Computes semantic similarity for all candidates in a single vectorized matrix multiplication."""
    # Ensure 2D shapes for multiplication
    if len(jd_vec.shape) == 1:
        jd_vec = jd_vec.reshape(1, -1)
    if len(cand_vecs.shape) == 1:
        cand_vecs = cand_vecs.reshape(1, -1)
        
    scores = (cand_vecs @ jd_vec.T).squeeze()
    scores = np.clip(scores, 0.0, 1.0)
    return np.atleast_1d(scores)

def get_C_map(jd_vec: np.ndarray, cand_vecs: np.ndarray, cand_ids: list) -> dict[str, float]:
    """Computes similarity and returns a dictionary mapping candidate IDs to embedding scores."""
    scores = compute_C_all(jd_vec, cand_vecs)
    return {str(cid): float(score) for cid, score in zip(cand_ids, scores)}
