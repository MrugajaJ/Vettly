
<div align="center">

```
██╗   ██╗███████╗████████╗████████╗██╗  ██╗   ██╗
██║   ██║██╔════╝╚══██╔══╝╚══██╔══╝██║  ╚██╗ ██╔╝
██║   ██║█████╗     ██║      ██║   ██║   ╚████╔╝ 
╚██╗ ██╔╝██╔══╝     ██║      ██║   ██║    ╚██╔╝  
 ╚████╔╝ ███████╗   ██║      ██║   ███████╗██║   
  ╚═══╝  ╚══════╝   ╚═╝      ╚═╝   ╚══════╝╚═╝   
```

**The AI Brain for Modern Hiring**

*Transforming 100,000 candidate profiles into a precise, behaviorally validated shortlist — in under 65 seconds.*

---

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-CPU-FF6F00?style=for-the-badge&logo=meta&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-MiniLM-blueviolet?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![RapidFuzz](https://img.shields.io/badge/RapidFuzz-Skill%20Matching-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## 📌 The Problem

> *"Recruiters are tasked with finding the perfect fit from oceans of profiles, but traditional keyword filters are simply not cutting it. They miss the hidden gems — candidates whose true potential, intent, and subtle behavioral signals are lost in the noise."*

Traditional ATS and keyword matching tools are fundamentally broken:

- ❌ **Keyword stuffing** lets unqualified candidates rank above genuine experts
- ❌ **Static resume scanning** ignores real-time candidate intent and availability
- ❌ **Binary skill checklists** treat self-declared expertise as ground truth
- ❌ **No fraud detection** leaves pipelines polluted with ghost and synthetic profiles
- ❌ **No behavioral signals** result in shortlists full of candidates who never respond

---

## 💡 The Solution: Vettly

**Vettly** is a **predictive candidate discovery and ranking engine** that combines semantic embeddings, verified skill trust scores, and real-time behavioral intent signals to rank candidates far beyond what keywords alone can achieve.

### Core Capabilities

| Capability | Description |
|---|---|
| 🧠 **Semantic Understanding** | Dense vector embeddings capture context, synonyms & narrative coherence |
| 🔬 **Verified Skill Trust** | Evidence-based competency scores using endorsements, tests & duration |
| 📡 **Behavioral Recruitability** | Real-time signals like activity recency, response rates & salary fit |
| 🛡️ **Fraud Defense** | Honeypot detection eliminates synthetic and spam profiles |
| ⚡ **Scale Performance** | 100,000 candidates ranked in ~64 seconds on a single CPU |
| 📋 **Explainable Output** | Deterministic, human-readable reasoning for every shortlisted candidate |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│           candidates.jsonl  +  job_description.json         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               STREAM PARSER (Two-Pass)                       │
│  Pass 1: Collect titles → Fit TF-IDF Vectorizer             │
│  Pass 2: Stream line-by-line → Apply disqualification       │
└──────────────────┬──────────────────┬───────────────────────┘
                   │                  │
              [KILLED]           [SURVIVORS]
           (Score = 0)                │
                                      ▼
┌─────────────────────────────────────────────────────────────┐
│            STAGE 1: HARD DISQUALIFY FILTER                  │
│  ✗ Profile completeness < 30                                │
│  ✗ Unverified email                                         │
│  ✗ Interview completion rate < 20%                          │
│  ✗ Inactive > 180 days                                      │
│  ✗ Not open to work                                         │
│  ✗ Zero industry overlap                                    │
│  ✗ Title TF-IDF similarity < 0.05                           │
│  ✗ Honeypot pattern detected                                │
└──────────────────────────┬──────────────────────────────────┘
                           │ ~60% survive
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         STAGE 2: SEMANTIC EMBEDDING SUBSYSTEM               │
│     SentenceTransformer (all-MiniLM-L6-v2, 384-dim)        │
│     FAISS IndexFlatIP → Cosine Similarity (C Score)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌──────────────┬────────────────────────┬─────────────────────┐
│  COMPONENT A │      COMPONENT B       │     COMPONENT C     │
│ Career Fit   │   Skill Trust Score    │ Semantic Similarity │
│   (40%)      │       (35%)            │      (25%)          │
│              │                        │                     │
│ TF-IDF title │ Fuzzy match (≥85%)     │ FAISS dot product   │
│ Jaccard ind. │ Proficiency weight     │ L2-normalized vecs  │
│ Decay KW     │ Endorsements           │                     │
│ YoE ratio    │ Assessment scores      │                     │
│              │ Usage duration         │                     │
└──────┬───────┴───────────┬────────────┴──────────┬──────────┘
       └───────────────────┴───────────────────────┘
                           │
              Raw Score = 0.40×A + 0.35×B + 0.25×C
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         STAGE 3: BEHAVIORAL MULTIPLIERS                     │
│  Final Score = Raw × Availability Mult × Location Mult      │
│  Availability clamp: [0.50, 1.25]                           │
│  Location clamp:     [0.70, 1.05]                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         STAGE 4: SORT + EXPLAIN + OUTPUT                    │
│  Sort: final_score → completeness → recruiter_saves         │
│  Top-100 + deterministic reasoning string per candidate     │
│  Output: submission.csv                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📐 Scoring Formulas

### Raw Score
```
Raw Score = 0.40 × A  +  0.35 × B  +  0.25 × C
```

### Component A — Career & Role Fit
```
A = 0.35 × title_sim
  + 0.25 × industry_match       (Jaccard overlap)
  + 0.25 × keyword_density      (exp(-0.15 × age_years) decay)
  + 0.15 × yoe_score            (min(YoE / JD.min_yoe, 1.0))
```

### Component B — Skill Trust Score
```
For each matched skill:
  trust = proficiency_weight × (0.35 + 0.25×endorse_w + 0.25×assess_w + 0.15×duration_w)

Proficiency weights: Beginner=0.40 | Intermediate=0.70 | Advanced=0.90 | Expert=1.00

B = min(0.75 × must_have_coverage + 0.25 × nice_coverage + cert_bonus, 1.0)
cert_bonus: +0.05 per matched cert, max 0.15
```

### Component C — Semantic Similarity
```
C = dot(L2_norm(jd_vec), L2_norm(cand_vec))   ∈ [0, 1]
  = cosine similarity via FAISS IndexFlatIP
```

### Final Score
```
Final Score = min(Raw Score × availability_mult × location_mult, 1.0)
```

### Availability Multiplier
```
Base: 1.0
  + 0.10  open_to_work == True
  + 0.10  active ≤ 14 days
  + 0.05  active ≤ 7 days (stacks)
  + 0.05  recruiter_response_rate ≥ 0.70
  + 0.05  offer_acceptance_rate ≥ 0.80
  - 0.05  avg_response_time > 72 hrs
  - 0.10  notice_period > 90 days
  - 0.15  interview_completion_rate < 0.50
  - 0.25  inactive > 90 days
  - 0.20  salary_min > JD budget_max
Clamp: [0.50, 1.25]
```

---

## 📁 Project Structure

```
Vettly/
├── data/
│   ├── candidates.jsonl           # 100K candidate profiles (input)
│   ├── job_description.json       # Structured JD (input)
│   └── precomputed/
│       ├── jd_vec.npy             # JD dense vector
│       ├── cand_vecs.npy          # Candidate dense vectors
│       ├── cand_ids.json          # ID → index mapping
│       ├── faiss.index            # FAISS flat index
│       └── tfidf.pkl              # Fitted TF-IDF vectorizer
├── src/
│   ├── precompute.py              # Stage 0: Offline embedding & indexing
│   ├── hard_filter.py             # Stage 1: 8-rule disqualification engine
│   ├── score_career.py            # Stage 2A: Title, industry, keyword, YoE
│   ├── score_skills.py            # Stage 2B: Fuzzy skill trust scoring
│   ├── score_embed.py             # Stage 2C: FAISS cosine similarity
│   ├── raw_score.py               # Weighted A+B+C combination
│   ├── availability.py            # Stage 3: Behavioral multipliers
│   ├── output.py                  # Stage 4: Sort, reason, export CSV
│   ├── pipeline.py                # Standard pipeline runner
│   └── run_fast_pipeline.py       # High-performance streaming runner
├── tests/
│   └── test_*.py                  # Unit tests per component
├── submission.csv                 # Final ranked output
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or 3.11 (recommended)
- 4 GB+ RAM (16 GB for full 100K run)
- CPU with AVX2 support for FAISS (most modern machines)

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install packages
pip install -r requirements.txt
```

**`requirements.txt`**
```
sentence-transformers==2.7.0
faiss-cpu==1.8.0
scikit-learn==1.4.2
numpy==1.26.4
pandas==2.2.2
rapidfuzz==3.9.3
python-dateutil==2.9.0
tqdm==4.66.4
polars
pytest==8.2.0
```

---

## 🚀 Running the Pipeline

### Step 1: Prepare Inputs

Ensure the following files exist:
```
data/candidates.jsonl        # One candidate JSON object per line
data/job_description.json    # Structured JD file
```

**Example `job_description.json`:**
```json
{
  "title": "Senior AI Engineer",
  "min_yoe": 5,
  "target_industries": ["Technology", "AI", "SaaS"],
  "preferred_locations": ["Bangalore", "Hyderabad", "Remote"],
  "budget_max_inr_lpa": 40,
  "must_have_skills": ["Python", "FAISS", "Embeddings", "Sentence Transformers"],
  "nice_to_have_skills": ["LLM Fine-tuning", "LoRA", "XGBoost"],
  "keywords": ["embeddings", "retrieval", "ranking", "vector database"],
  "description": "Full JD text here..."
}
```

### Step 2: Run the Fast Pipeline

```bash
python -m src.run_fast_pipeline
```

### Step 3: Check Output

```bash
# View top 10 results
python -c "import pandas as pd; print(pd.read_csv('submission.csv')[['rank','candidate_id','final_score','reasoning']].head(10).to_string())"
```

---

## 📊 Output Format

The final `submission.csv` contains 100 rows with the following schema:

| Column | Type | Description |
|---|---|---|
| `rank` | int | 1–100, sorted by final score |
| `candidate_id` | string | Format: `CAND_XXXXXXX` |
| `final_score` | float | Composite score ∈ [0, 1] |
| `raw_score` | float | Pre-multiplier score |
| `component_scores` | string | `A=0.XXX\|B=0.XXX\|C=0.XXX` |
| `availability_mult` | float | Behavioral multiplier applied |
| `location_mult` | float | Location multiplier applied |
| `reasoning` | string | Human-readable match explanation |

**Sample reasoning output:**
```
Arjun Khanna | 6.7y exp | Lead AI Engineer @ Razorpay |
Skill match: 20% must-haves covered | Top skills: Information Retrieval, Vector Search, Machine Learning |
Open to work: True | Notice: 30d | GitHub: 33.7 |
Score: 0.5565 (A=0.547 B=0.184 C=0.803)
```

---

## ⚡ Performance Metrics

| Stage | Operation | Time |
|---|---|---|
| Load + TF-IDF Fit | Stream 100K titles, fit vectorizer | ~5 sec |
| Stage 1 Hard Filter | 8-rule disqualification stream | ~8 sec |
| Stage 2C Embeddings | SentenceTransformer + FAISS | ~30 sec |
| Stage 2A Career Score | TF-IDF + decay per survivor | ~10 sec |
| Stage 2B Skill Trust | Fuzzy match per survivor | ~6 sec |
| Stage 3 Multipliers | Behavioral scoring | ~3 sec |
| Stage 4 Sort + CSV | Rank + explain + write | ~2 sec |
| **TOTAL** | **Full pipeline, 100K candidates** | **~64 sec** |

- **Memory usage:** `< 150 MB` at peak
- **Hardware:** Single CPU, no GPU required
- **Hard filter elimination rate:** ~40% of candidates pruned before embedding

---

## 🛡️ Fraud & Quality Defense

Vettly implements a multi-layer defense against low-quality data:

**Hard Kills (Score = 0):**
- Profile completeness below 30%
- Unverified email address
- Interview completion rate below 20%
- Inactive for more than 180 days
- Not marked open to work
- Zero industry overlap with JD
- Title similarity below 5% (TF-IDF cosine)
- **Honeypot pattern:** GitHub = -1 + no assessments + 0 endorsements + <5 connections

**Soft Penalties (Multiplier Reduction):**
- Inactive 90+ days: −25%
- Interview completion rate <50%: −15%
- Salary expectations exceed budget: −20%
- Notice period > 90 days: −10%

---

## 🔬 Technology Stack

| Technology | Role | Why Selected |
|---|---|---|
| `sentence-transformers` | Dense semantic embeddings | Captures context beyond keywords; fast on CPU |
| `faiss-cpu` | Vector similarity search | C++-optimized; sub-second cosine for 100K vectors |
| `scikit-learn` | TF-IDF vectorization | Efficient sparse matrix ops for lexical matching |
| `rapidfuzz` | Fuzzy skill matching | C++ speed; handles skill name variations reliably |
| `python-dateutil` | Date parsing | Robust timezone-aware temporal calculations |
| `pandas` / `polars` | Data wrangling & CSV output | High-performance tabular operations |
| `numpy` | Matrix math & decay calc | Vectorized exponential operations |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Hard filter kill rules (staleness, honeypot, completeness)
- Skill trust score boundary conditions (max = 1.0, no skills = 0.0)
- Career score decay accuracy
- Multiplier clamping behavior
- CSV validation (100 rows, valid IDs, scores in [0,1])

---

## 📈 Submission Validation

```python
import pandas as pd

df = pd.read_csv('submission.csv')

assert len(df) <= 100, "Too many rows"
assert df['rank'].tolist() == list(range(1, len(df)+1)), "Rank gaps detected"
assert df['candidate_id'].str.match(r'^CAND_[0-9]{7}$').all(), "Invalid ID format"
assert (df['final_score'] >= 0).all() and (df['final_score'] <= 1).all(), "Score out of range"

print("✅ All validation checks passed.")
print(df['final_score'].describe())
```

---

## 🔮 Advanced Upgrades (Optional)

| Feature | Description |
|---|---|
| **Reciprocal Rank Fusion (RRF)** | Fuse BM25 + embedding ranks: `1/(k+rank_bm25) + 1/(k+rank_embed)` |
| **LambdaMART LTR** | XGBoost ranker trained on hired/rejected labels using [A, B, C, signals] as features |
| **Education Tier Bonus** | Add prestige signal: tier_1 = +0.08, tier_2 = +0.04 to Career Score |
| **Continuous Salary Overlap** | Replace binary penalty with overlap ratio as a soft signal |

---

<div align="center">

---

*Built for the Redrob AI Challenge — India Runs Hackathon*

**Vettly** — *Find the signal. Ignore the noise.*

</div>
