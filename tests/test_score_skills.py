import pytest
from src.score_skills import compute_B

@pytest.fixture
def base_jd():
    return {
        "must_have_skills": ["Python", "PyTorch", "LLMs"],
        "nice_to_have_skills": ["Docker", "AWS"]
    }

def test_compute_B_capped_at_one(base_jd):
    # Candidate with perfect skills, max endorsements, max assessments, max duration
    perfect_candidate = {
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 100, "duration_months": 48},
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 100, "duration_months": 48},
            {"name": "LLMs", "proficiency": "expert", "endorsements": 100, "duration_months": 48},
            {"name": "Docker", "proficiency": "expert", "endorsements": 100, "duration_months": 48},
            {"name": "AWS", "proficiency": "expert", "endorsements": 100, "duration_months": 48}
        ],
        "skill_assessment_scores": {
            "Python": 100,
            "PyTorch": 100,
            "LLMs": 100,
            "Docker": 100,
            "AWS": 100
        },
        "certifications": ["AWS Certified Architect", "Docker Certified Associate", "Python Master"]
    }
    
    res = compute_B(perfect_candidate, base_jd)
    assert res["B"] <= 1.0
    # Perfect candidate should achieve the maximum score of 1.0
    assert res["B"] == 1.0

def test_compute_B_empty_skills(base_jd):
    empty_candidate = {
        "skills": [],
        "skill_assessment_scores": {},
        "certifications": []
    }
    
    res = compute_B(empty_candidate, base_jd)
    assert res["B"] == 0.0
    assert res["must_have_coverage"] == 0.0
    assert res["nice_coverage"] == 0.0
