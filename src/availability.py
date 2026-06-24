import datetime
from dateutil import parser

def compute_multipliers(candidate: dict, jd: dict) -> dict[str, float]:
    """Computes the availability and location multipliers for a candidate, supporting nested schemas."""
    signals = candidate.get("redrob_signals") or {}
    profile = candidate.get("profile") or {}
    
    # --- Availability Multiplier ---
    availability_mult = 1.0
    
    # open_to_work_flag
    open_to_work = signals.get("open_to_work_flag")
    if open_to_work is None:
        open_to_work = candidate.get("open_to_work_flag")
    if open_to_work is True:
        availability_mult += 0.10
        
    # last_active_date
    last_active_str = signals.get("last_active_date") or candidate.get("last_active_date")
    days_inactive = None
    if last_active_str:
        try:
            last_active = parser.parse(last_active_str)
            if last_active.tzinfo is not None:
                today = datetime.datetime.now(datetime.timezone.utc)
            else:
                today = datetime.datetime.now()
            days_inactive = (today - last_active).days
        except Exception:
            pass
            
    if days_inactive is not None:
        if days_inactive <= 14:
            availability_mult += 0.10
            if days_inactive <= 7:
                availability_mult += 0.05  # stacks
        if days_inactive > 90:
            availability_mult -= 0.25
            
    # recruiter_response_rate
    response_rate = signals.get("recruiter_response_rate")
    if response_rate is None:
        response_rate = candidate.get("recruiter_response_rate", 0.0)
    if float(response_rate) >= 0.70:
        availability_mult += 0.05
        
    # offer_acceptance_rate
    acceptance_rate = signals.get("offer_acceptance_rate")
    if acceptance_rate is None:
        acceptance_rate = candidate.get("offer_acceptance_rate")
    if acceptance_rate is not None and acceptance_rate != -1:
        if float(acceptance_rate) >= 0.80:
            availability_mult += 0.05
            
    # avg_response_time_hours
    avg_resp_time = signals.get("avg_response_time_hours")
    if avg_resp_time is None:
        avg_resp_time = candidate.get("avg_response_time_hours")
    if avg_resp_time is not None and float(avg_resp_time) > 72:
        availability_mult -= 0.05
        
    # notice_period_days
    notice_period = signals.get("notice_period_days")
    if notice_period is None:
        notice_period = candidate.get("notice_period_days")
    if notice_period is not None and int(notice_period) > 90:
        availability_mult -= 0.10
        
    # interview_completion_rate
    completion_rate = signals.get("interview_completion_rate")
    if completion_rate is None:
        completion_rate = candidate.get("interview_completion_rate", 0.0)
    if float(completion_rate) < 0.50:
        availability_mult -= 0.15
        
    # expected_salary_range_inr_lpa.min
    salary_range = signals.get("expected_salary_range_inr_lpa") or candidate.get("expected_salary_range_inr_lpa") or {}
    salary_min = 0.0
    if isinstance(salary_range, dict):
        salary_min = salary_range.get("min") or 0.0
    elif isinstance(salary_range, (int, float)):
        salary_min = salary_range
        
    budget_max = jd.get("budget_max_inr_lpa") or 40
    if salary_min > budget_max:
        availability_mult -= 0.20
        
    # Clamp availability_mult to [0.50, 1.25]
    availability_mult = max(0.50, min(availability_mult, 1.25))
    
    # --- Location Multiplier ---
    location_mult = 1.0
    
    cand_loc = str(profile.get("location") or candidate.get("location") or "").lower().strip()
    jd_locs = {str(loc).lower().strip() for loc in (jd.get("preferred_locations") or []) if loc}
    
    if cand_loc in jd_locs:
        location_mult += 0.05
    else:
        willing_to_relocate = signals.get("willing_to_relocate")
        if willing_to_relocate is None:
            willing_to_relocate = candidate.get("willing_to_relocate", True)
        if willing_to_relocate is False:
            location_mult -= 0.05
            
    # Clamp location_mult to [0.70, 1.05]
    location_mult = max(0.70, min(location_mult, 1.05))
    
    return {
        "availability_mult": round(availability_mult, 4),
        "location_mult": round(location_mult, 4)
    }

def apply_multipliers(scored_results: list[dict], jd: dict) -> list[dict]:
    """Applies availability and location multipliers to the scored results."""
    updated_results = []
    for res in scored_results:
        mults = compute_multipliers(res["candidate"], jd)
        av_mult = mults["availability_mult"]
        loc_mult = mults["location_mult"]
        
        raw_score = res["raw_score"]
        final_score = round(min(raw_score * av_mult * loc_mult, 1.0), 4)
        
        updated_res = res.copy()
        updated_res["availability_mult"] = av_mult
        updated_res["location_mult"] = loc_mult
        updated_res["final_score"] = final_score
        updated_results.append(updated_res)
        
    return updated_results
