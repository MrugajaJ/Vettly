import datetime
from dateutil import parser
from sklearn.metrics.pairwise import cosine_similarity

def is_killed(candidate: dict, jd: dict, tfidf) -> tuple[bool, str]:
    """Applies the 8 hard filtering rules in order, short-circuiting on first match.
    
    Supports both nested (actual dataset) and flat (test fixture) candidate structures.
    """
    signals = candidate.get("redrob_signals") or {}
    profile = candidate.get("profile") or {}
    
    # 1. Profile completeness score < 30
    profile_completeness = signals.get("profile_completeness_score") or candidate.get("profile_completeness_score", 0)
    if profile_completeness < 30:
        return True, f"profile_completeness_score {profile_completeness} < 30"
        
    # 2. Verified email == False
    verified_email = signals.get("verified_email")
    if verified_email is None:
        verified_email = candidate.get("verified_email", False)
    if verified_email is False:
        return True, "verified_email == False"
        
    # 3. Interview completion rate < 0.20
    interview_completion_rate = signals.get("interview_completion_rate")
    if interview_completion_rate is None:
        interview_completion_rate = candidate.get("interview_completion_rate", 0.0)
    if interview_completion_rate < 0.20:
        return True, f"interview_completion_rate {interview_completion_rate} < 0.20"
        
    # 4. (today - last_active_date).days > 180
    last_active_date_str = signals.get("last_active_date") or candidate.get("last_active_date")
    if not last_active_date_str:
        return True, "Missing last_active_date"
    try:
        last_active_date = parser.parse(last_active_date_str)
        if last_active_date.tzinfo is not None:
            today = datetime.datetime.now(datetime.timezone.utc)
        else:
            today = datetime.datetime.now()
        days_inactive = (today - last_active_date).days
        if days_inactive > 180:
            return True, f"Inactive for {days_inactive} days (> 180 days)"
    except Exception as e:
        return True, f"Failed to parse last_active_date: {str(e)}"
        
    # 5. Open to work flag == False
    open_to_work = signals.get("open_to_work_flag")
    if open_to_work is None:
        open_to_work = candidate.get("open_to_work_flag", False)
    if open_to_work is False:
        return True, "open_to_work_flag == False"
        
    # 6. Zero overlap between candidate's industries and jd.target_industries using Python set intersection
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
    # Fallback to top-level industries list if present (like in mock fixtures)
    for ind in (candidate.get("industries") or []):
        if ind:
            cand_industries.add(ind.strip().lower())
            
    jd_industries = {ind.strip().lower() for ind in (jd.get("target_industries") or []) if ind}
    if not cand_industries.intersection(jd_industries):
        return True, f"Zero industry overlap (Candidate: {cand_industries}, JD: {jd_industries})"
        
    # 7. TF-IDF cosine similarity between current_title and jd.title < 0.05
    cand_title = profile.get("current_title") or candidate.get("current_title", "")
    jd_title = jd.get("title", "")
    if not cand_title or not jd_title:
        sim = 0.0
    else:
        try:
            cand_tfidf = tfidf.transform([cand_title])
            jd_tfidf = tfidf.transform([jd_title])
            sim = cosine_similarity(cand_tfidf, jd_tfidf)[0][0]
        except Exception:
            sim = 0.0
    if sim < 0.05:
        return True, f"Title similarity {sim:.4f} < 0.05 (Candidate: '{cand_title}', JD: '{jd_title}')"
        
    # 8. Honeypot check: github_activity_score == -1 AND empty skill_assessment_scores AND endorsements_received == 0 AND connection_count < 5
    github_score = signals.get("github_activity_score")
    if github_score is None:
        github_score = candidate.get("github_activity_score", 0)
        
    skill_assessments = signals.get("skill_assessment_scores")
    if skill_assessments is None:
        skill_assessments = candidate.get("skill_assessment_scores") or {}
        
    endorsements = signals.get("endorsements_received")
    if endorsements is None:
        endorsements = candidate.get("endorsements_received", 0)
        
    connections = signals.get("connection_count")
    if connections is None:
        connections = candidate.get("connection_count", 0)
    
    if (github_score == -1 and 
        not skill_assessments and 
        endorsements == 0 and 
        connections < 5):
        return True, f"Honeypot detected (GitHub: {github_score}, Assessments: {skill_assessments}, Endorsements: {endorsements}, Connections: {connections})"
        
    return False, ""

def apply_hard_filter(candidates: list[dict], jd: dict, tfidf) -> tuple[list[dict], list[dict]]:
    """Applies the hard filter rules to a list of candidates.
    
    Returns (survivors_list, killed_list).
    """
    survivors = []
    killed = []
    for cand in candidates:
        killed_flag, reason = is_killed(cand, jd, tfidf)
        if killed_flag:
            killed.append({
                "candidate_id": cand.get("candidate_id") or cand.get("id"),
                "candidate": cand,
                "reason": reason
            })
        else:
            survivors.append(cand)
    return survivors, killed
