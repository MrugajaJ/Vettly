from rapidfuzz import fuzz

def fuzzy_match_skill(cand_skill_name: str, jd_skills: list[str]) -> str | None:
    """Finds the best matching JD skill for a candidate's skill using a threshold of 85.
    
    Returns the JD skill name if matched, else None.
    """
    if not cand_skill_name or not jd_skills:
        return None
    best_ratio = 0
    best_skill = None
    for jd_s in jd_skills:
        ratio = fuzz.token_sort_ratio(cand_skill_name.lower().strip(), jd_s.lower().strip())
        if ratio > best_ratio:
            best_ratio = ratio
            best_skill = jd_s
    if best_ratio >= 85:
        return best_skill
    return None

def skill_trust(skill_obj, assess_scores: dict, jd_skill: str) -> float:
    """Calculates the trust score for a matched skill.
    
    Formula: trust = prof_weight * (0.35 + 0.25 * endorse_w + 0.25 * assess_w + 0.15 * duration_w)
    """
    if isinstance(skill_obj, str):
        name = skill_obj
        proficiency = "intermediate"
        endorsements = 0
        duration_months = 0
    elif isinstance(skill_obj, dict):
        name = skill_obj.get("name") or ""
        proficiency = skill_obj.get("proficiency") or "intermediate"
        endorsements = skill_obj.get("endorsements") or 0
        duration_months = skill_obj.get("duration_months") or 0
    else:
        return 0.0
        
    # 1. Proficiency Weight
    prof = str(proficiency).lower().strip()
    if prof == "beginner":
        prof_weight = 0.40
    elif prof == "intermediate":
        prof_weight = 0.70
    elif prof == "advanced":
        prof_weight = 0.90
    elif prof in ["expert", "master"]:
        prof_weight = 1.00
    else:
        prof_weight = 0.70  # default
        
    # 2. Endorsement Weight
    endorse_w = min(float(endorsements) / 20.0, 1.0)
    
    # 3. Assessment Weight
    assess_val = 0.0
    if isinstance(assess_scores, dict):
        # Try exact key lookup or fuzzy key lookup in the assessment scores
        assess_val = assess_scores.get(jd_skill) or assess_scores.get(name) or 0.0
    assess_w = float(assess_val) / 100.0
    
    # 4. Duration Weight
    duration_w = min(float(duration_months) / 24.0, 1.0)
    
    trust = prof_weight * (0.35 + 0.25 * endorse_w + 0.25 * assess_w + 0.15 * duration_w)
    return float(trust)

def compute_B(candidate: dict, jd: dict) -> dict:
    """Computes the skill trust score (B) for a candidate.
    
    Formula: B = min(0.75 * must_cov + 0.25 * nice_cov + cert_bonus, 1.0)
    """
    must_have_skills = jd.get("must_have_skills") or []
    nice_to_have_skills = jd.get("nice_to_have_skills") or []
    
    must_trust = {s: 0.0 for s in must_have_skills}
    nice_trust = {s: 0.0 for s in nice_to_have_skills}
    
    cand_skills = candidate.get("skills") or []
    assess_scores = candidate.get("skill_assessment_scores") or {}
    
    # Calculate trust for matching skills
    for s_obj in cand_skills:
        s_name = s_obj if isinstance(s_obj, str) else s_obj.get("name", "")
        if not s_name:
            continue
            
        matched_must = fuzzy_match_skill(s_name, must_have_skills)
        if matched_must:
            t = skill_trust(s_obj, assess_scores, matched_must)
            must_trust[matched_must] = max(must_trust[matched_must], t)
            
        matched_nice = fuzzy_match_skill(s_name, nice_to_have_skills)
        if matched_nice:
            t = skill_trust(s_obj, assess_scores, matched_nice)
            nice_trust[matched_nice] = max(nice_trust[matched_nice], t)
            
    must_cov = sum(must_trust.values()) / len(must_have_skills) if must_have_skills else 0.0
    nice_cov = sum(nice_trust.values()) / len(nice_to_have_skills) if nice_to_have_skills else 0.0
    
    # 5. Certification Bonus
    certs = candidate.get("certifications") or candidate.get("certs") or []
    all_jd_skills = must_have_skills + nice_to_have_skills
    cert_matches = 0
    for cert in certs:
        cert_name = cert if isinstance(cert, str) else cert.get("name", "")
        if not cert_name:
            continue
            
        # Check if cert fuzzy matches any JD skill or contains it as a substring
        matched = False
        for jd_s in all_jd_skills:
            if fuzz.token_sort_ratio(cert_name.lower().strip(), jd_s.lower().strip()) >= 85:
                matched = True
                break
            if jd_s.lower().strip() in cert_name.lower():
                matched = True
                break
        if matched:
            cert_matches += 1
            
    cert_bonus = min(cert_matches * 0.05, 0.15)
    
    B = min(0.75 * must_cov + 0.25 * nice_cov + cert_bonus, 1.0)
    
    return {
        "B": round(B, 4),
        "must_have_coverage": round(must_cov, 4),
        "nice_coverage": round(nice_cov, 4),
        "cert_bonus": round(cert_bonus, 4)
    }
