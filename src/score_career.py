import datetime
import numpy as np
from dateutil import parser
from sklearn.metrics.pairwise import cosine_similarity

def compute_job_age(job: dict) -> float:
    """Calculate the age of a job in years from today."""
    is_current = job.get("is_current")
    end_date_str = job.get("end_date")
    if is_current or not end_date_str or str(end_date_str).lower().strip() in ["present", "current", "none", "null"]:
        return 0.0
        
    try:
        end_date = parser.parse(str(end_date_str))
        if end_date.tzinfo is not None:
            today = datetime.datetime.now(datetime.timezone.utc)
        else:
            today = datetime.datetime.now()
        days = (today - end_date).days
        return max(0.0, days / 365.25)
    except Exception:
        return 0.0

def compute_raw_keyword_score(candidate: dict, jd: dict) -> float:
    """Compute the raw, recency-decayed keyword score for a candidate, supporting nested career_history."""
    keywords = jd.get("keywords") or []
    history = candidate.get("career_history") or candidate.get("experience") or candidate.get("work_experience") or []
    if not keywords or not history:
        return 0.0
        
    total_score = 0.0
    for job in history:
        if not isinstance(job, dict):
            continue
        job_title = job.get("title") or ""
        job_desc = job.get("description") or ""
        job_text = f"{job_title} {job_desc}".lower()
        
        # Count keyword occurrences
        kw_count = sum(job_text.count(str(kw).lower()) for kw in keywords)
        
        age = compute_job_age(job)
        decay = np.exp(-0.15 * age)
        total_score += kw_count * decay
        
    return float(total_score)

def compute_keyword_max(candidates: list[dict], jd: dict, tfidf=None) -> float:
    """Scans the candidate pool and returns the maximum raw keyword score."""
    max_val = 0.0
    for cand in candidates:
        score = compute_raw_keyword_score(cand, jd)
        if score > max_val:
            max_val = score
    # Return at least 1.0 to avoid division by zero
    return max(max_val, 1.0)

def compute_A(candidate: dict, jd: dict, tfidf, keyword_max: float) -> dict:
    """Computes the career fit score (A) for a candidate, supporting nested and flat schemas."""
    profile = candidate.get("profile") or {}
    
    # 1. Title Similarity
    cand_title = profile.get("current_title") or candidate.get("current_title", "")
    jd_title = jd.get("title", "")
    if not cand_title or not jd_title:
        title_sim = 0.0
    else:
        try:
            cand_tfidf = tfidf.transform([cand_title])
            jd_tfidf = tfidf.transform([jd_title])
            title_sim = float(cosine_similarity(cand_tfidf, jd_tfidf)[0][0])
        except Exception:
            title_sim = 0.0
            
    # 2. Industry Match (Jaccard similarity)
    cand_industries = set()
    # Check profile industry
    prof_ind = profile.get("current_industry")
    if prof_ind:
        cand_industries.add(prof_ind.strip().lower())
    # Check career history industries
    history = candidate.get("career_history") or candidate.get("experience") or candidate.get("work_experience") or []
    for job in history:
        if isinstance(job, dict) and job.get("industry"):
            cand_industries.add(job["industry"].strip().lower())
    # Fallback to top-level industries list if present
    for ind in (candidate.get("industries") or []):
        if ind:
            cand_industries.add(ind.strip().lower())
            
    jd_industries = {ind.strip().lower() for ind in (jd.get("target_industries") or []) if ind}
    if not jd_industries:
        industry_match = 0.0
    else:
        union = cand_industries.union(jd_industries)
        intersection = cand_industries.intersection(jd_industries)
        industry_match = len(intersection) / len(union) if union else 0.0
        
    # 3. Keyword Density
    raw_kw = compute_raw_keyword_score(candidate, jd)
    prod_keyword_density = min(raw_kw / keyword_max, 1.0)
    
    # 4. YoE Score
    yoe = profile.get("years_of_experience") or profile.get("yoe") or candidate.get("years_of_experience") or candidate.get("yoe") or 0.0
    min_yoe = jd.get("min_yoe") or 5
    if min_yoe <= 0:
        yoe_score = 1.0
    else:
        yoe_score = min(float(yoe) / float(min_yoe), 1.0)
        
    A = 0.35 * title_sim + 0.25 * industry_match + 0.25 * prod_keyword_density + 0.15 * yoe_score
    
    return {
        "A": round(A, 4),
        "title_sim": round(title_sim, 4),
        "industry_match": round(industry_match, 4),
        "prod_keyword_density": round(prod_keyword_density, 4),
        "yoe_score": round(yoe_score, 4)
    }
