import polars as pl

def generate_reasoning(result: dict, jd: dict) -> str:
    """Generates a structured reasoning string for a candidate's fit, supporting nested schemas."""
    candidate = result["candidate"]
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    
    name = profile.get("anonymized_name") or profile.get("name") or candidate.get("name") or "Unknown"
    yoe = profile.get("years_of_experience") or profile.get("yoe") or candidate.get("years_of_experience") or candidate.get("yoe") or 0
    
    # Get current company/title
    history = candidate.get("career_history") or candidate.get("experience") or candidate.get("work_experience") or []
    company = profile.get("current_company") or "Unknown"
    title = profile.get("current_title") or candidate.get("current_title") or ""
    
    if history and isinstance(history, list) and isinstance(history[0], dict):
        if not title:
            title = history[0].get("title") or ""
        if company == "Unknown":
            company = history[0].get("company") or history[0].get("company_name") or "Unknown"
            
    if not title:
        title = "Candidate"
        
    must_have_coverage = result.get("must_have_coverage", 0.0)
    
    # Extract top 3 skill names
    cand_skills = candidate.get("skills") or []
    def skill_key(s):
        if isinstance(s, dict):
            return (s.get("endorsements") or 0) + (s.get("duration_months") or 0)
        return 0
    sorted_skills = sorted(cand_skills, key=skill_key, reverse=True)
    skill_names = []
    for s in sorted_skills[:3]:
        if isinstance(s, dict):
            skill_names.append(s.get("name", ""))
        elif isinstance(s, str):
            skill_names.append(s)
    top_skills_str = ", ".join(filter(None, skill_names)) or "None"
    
    open_to_work = bool(signals.get("open_to_work_flag") if signals.get("open_to_work_flag") is not None else candidate.get("open_to_work_flag", False))
    notice_period_days = signals.get("notice_period_days") if signals.get("notice_period_days") is not None else candidate.get("notice_period_days", 0)
    github_activity_score = signals.get("github_activity_score") if signals.get("github_activity_score") is not None else candidate.get("github_activity_score", 0)
    
    final_score = result["final_score"]
    A = result["A"]
    B = result["B"]
    C = result["C"]
    
    return (
        f"{name} | {yoe}y exp | {title} @ {company} | "
        f"Skill match: {must_have_coverage:.0%} must-haves covered | "
        f"Top skills: {top_skills_str} | Open to work: {open_to_work} | "
        f"Notice: {notice_period_days}d | GitHub: {github_activity_score} | "
        f"Score: {final_score:.4f} (A={A:.3f} B={B:.3f} C={C:.3f})"
    )

def write_submission(results: list[dict], jd: dict, out_path: str = 'submission.csv') -> pl.DataFrame:
    """Converts results to a Polars DataFrame, ranks the top 100, and writes them to a CSV."""
    rows = []
    for res in results:
        cand = res["candidate"]
        signals = cand.get("redrob_signals") or {}
        reasoning = generate_reasoning(res, jd)
        
        rows.append({
            "candidate_id": res["candidate_id"],
            "final_score": res["final_score"],
            "raw_score": res["raw_score"],
            "availability_mult": res["availability_mult"],
            "location_mult": res["location_mult"],
            "reasoning": reasoning,
            "profile_completeness_score": signals.get("profile_completeness_score") or cand.get("profile_completeness_score", 0),
            "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d") or cand.get("saved_by_recruiters_30d", 0),
            "component_scores": f"A={res['A']:.3f}, B={res['B']:.3f}, C={res['C']:.3f}"
        })
        
    df = pl.DataFrame(rows)
    
    # Sort by final_score, then profile_completeness_score, then saved_by_recruiters_30d (all descending)
    df_sorted = df.sort(
        by=["final_score", "profile_completeness_score", "saved_by_recruiters_30d"],
        descending=[True, True, True]
    )
    
    # Take top 100
    df_top100 = df_sorted.head(100)
    
    # Add rank column (1-indexed)
    df_top100 = df_top100.with_columns(
        pl.int_range(1, df_top100.height + 1).alias("rank")
    )
    
    # Select and reorder
    df_final = df_top100.select([
        "rank",
        "candidate_id",
        "final_score",
        "raw_score",
        "component_scores",
        "availability_mult",
        "location_mult",
        "reasoning"
    ])
    
    df_final.write_csv(out_path)
    print(f"Successfully wrote top {df_final.height} candidates to {out_path}")
    
    return df_final
