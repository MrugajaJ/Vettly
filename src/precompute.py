import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def build_candidate_text(cand: dict) -> str:
    """Build a consolidated text string from candidate fields.
    
    Structure: current_title + headline + all skill names joined + all career titles and descriptions capped at 500 chars each.
    """
    current_title = cand.get("current_title") or ""
    headline = cand.get("headline") or ""
    
    # Skill names
    skills = cand.get("skills") or []
    skill_names = []
    for s in skills:
        if isinstance(s, dict):
            skill_names.append(s.get("name") or "")
        elif isinstance(s, str):
            skill_names.append(s)
    skills_joined = " ".join(filter(None, skill_names))
    
    # Career history
    history = cand.get("career_history") or cand.get("experience") or cand.get("work_experience") or []
    history_parts = []
    if isinstance(history, list):
        for job in history:
            if isinstance(job, dict):
                title = job.get("title") or ""
                description = job.get("description") or ""
                title_capped = title[:500]
                description_capped = description[:500]
                if title_capped:
                    history_parts.append(title_capped)
                if description_capped:
                    history_parts.append(description_capped)
                    
    history_joined = " ".join(history_parts)
    
    # Combine parts
    parts = [current_title, headline, skills_joined, history_joined]
    cleaned_parts = [p.strip() for p in parts if p and p.strip()]
    return " ".join(cleaned_parts)

def build_jd_text(jd: dict) -> str:
    """Build a consolidated text string from job description fields."""
    parts = []
    if jd.get("title"):
        parts.append(jd["title"])
    if jd.get("must_have_skills"):
        parts.append(" ".join(jd["must_have_skills"]))
    if jd.get("nice_to_have_skills"):
        parts.append(" ".join(jd["nice_to_have_skills"]))
    if jd.get("keywords"):
        parts.append(" ".join(jd["keywords"]))
    if jd.get("description"):
        parts.append(jd["description"])
    cleaned_parts = [p.strip() for p in parts if p and p.strip()]
    return " ".join(cleaned_parts)

def precompute():
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vettly_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(vettly_dir, "data")
    precomputed_dir = os.path.join(data_dir, "precomputed")
    
    os.makedirs(precomputed_dir, exist_ok=True)
    
    candidates_path = os.path.join(data_dir, "candidates.json")
    jd_path = os.path.join(data_dir, "job_description.json")
    
    print(f"Loading job description from {jd_path}...")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_data = json.load(f)
    jd_text = build_jd_text(jd_data)
    
    print(f"Loading candidates from {candidates_path}...")
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(f"Candidates file not found at {candidates_path}")
        
    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
        
    print(f"Processing {len(candidates)} candidates...")
    candidate_texts = []
    candidate_ids = []
    for cand in tqdm(candidates, desc="Building candidate texts"):
        cand_id = cand.get("id") or cand.get("candidate_id") or ""
        candidate_ids.append(str(cand_id))
        candidate_texts.append(build_candidate_text(cand))
        
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Embedding job description...")
    jd_vec = model.encode(jd_text, normalize_embeddings=True).astype("float32")
    
    print("Embedding candidates...")
    cand_vecs = model.encode(
        candidate_texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")
    
    # FAISS index
    print("Building FAISS index...")
    dimension = 384
    index = faiss.IndexFlatIP(dimension)
    index.add(cand_vecs)
    
    # TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    tfidf.fit(candidate_texts)
    
    # Save outputs
    jd_vec_path = os.path.join(precomputed_dir, "jd_vec.npy")
    cand_vecs_path = os.path.join(precomputed_dir, "cand_vecs.npy")
    cand_ids_path = os.path.join(precomputed_dir, "cand_ids.json")
    faiss_index_path = os.path.join(precomputed_dir, "faiss.index")
    tfidf_pkl_path = os.path.join(precomputed_dir, "tfidf.pkl")
    
    print("Saving precomputed outputs...")
    np.save(jd_vec_path, jd_vec)
    np.save(cand_vecs_path, cand_vecs)
    
    with open(cand_ids_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f, indent=2)
        
    faiss.write_index(index, faiss_index_path)
    
    with open(tfidf_pkl_path, "wb") as f:
        pickle.dump(tfidf, f)
        
    print("Precomputation completed successfully.")

if __name__ == "__main__":
    precompute()
