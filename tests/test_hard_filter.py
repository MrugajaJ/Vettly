import datetime
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from src.hard_filter import is_killed

@pytest.fixture
def mock_tfidf():
    tfidf = TfidfVectorizer()
    tfidf.fit(["Senior AI Engineer", "Software Engineer", "Data Scientist", "Python Developer"])
    return tfidf

@pytest.fixture
def base_jd():
    return {
        "title": "Senior AI Engineer",
        "min_yoe": 5,
        "target_industries": ["Technology", "AI", "SaaS"],
        "budget_max_inr_lpa": 40
    }

@pytest.fixture
def valid_candidate():
    return {
        "id": "CAND_1234567",
        "name": "Jane Doe",
        "profile_completeness_score": 85,
        "verified_email": True,
        "interview_completion_rate": 0.90,
        "last_active_date": datetime.datetime.now().isoformat(),
        "open_to_work_flag": True,
        "industries": ["AI", "Technology"],
        "current_title": "Senior AI Engineer",
        "github_activity_score": 5,
        "skill_assessment_scores": {"Python": 90},
        "endorsements_received": 10,
        "connection_count": 100,
        "skills": [{"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 36}]
    }

def test_inactive_candidate_is_killed(valid_candidate, base_jd, mock_tfidf):
    # Set last_active_date to 200 days ago
    past_date = datetime.datetime.now() - datetime.timedelta(days=200)
    valid_candidate["last_active_date"] = past_date.isoformat()
    
    killed, reason = is_killed(valid_candidate, base_jd, mock_tfidf)
    assert killed is True
    assert "Inactive" in reason or "inactive" in reason.lower()

def test_honeypot_candidate_is_killed(valid_candidate, base_jd, mock_tfidf):
    # Modify candidate to match honeypot signature
    valid_candidate["github_activity_score"] = -1
    valid_candidate["skill_assessment_scores"] = {}
    valid_candidate["endorsements_received"] = 0
    valid_candidate["connection_count"] = 2
    
    killed, reason = is_killed(valid_candidate, base_jd, mock_tfidf)
    assert killed is True
    assert "honeypot" in reason.lower()

def test_valid_candidate_passes(valid_candidate, base_jd, mock_tfidf):
    killed, reason = is_killed(valid_candidate, base_jd, mock_tfidf)
    assert killed is False
    assert reason == ""
