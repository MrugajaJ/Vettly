
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

## The Problem

> *"Recruiters are tasked with finding the perfect fit from oceans of profiles, but traditional keyword filters are simply not cutting it. They miss the hidden gems — candidates whose true potential, intent, and subtle behavioral signals are lost in the noise."*

Traditional ATS and keyword matching tools are fundamentally broken:

- **Keyword stuffing** lets unqualified candidates rank above genuine experts
- **Static resume scanning** ignores real-time candidate intent and availability
- **Binary skill checklists** treat self-declared expertise as ground truth
- **No fraud detection** leaves pipelines polluted with ghost and synthetic profiles
- **No behavioral signals** result in shortlists full of candidates who never respond

---

## The Solution: Vettly

**Vettly** is a **predictive candidate discovery and ranking engine** that combines semantic embeddings, verified skill trust scores, and real-time behavioral intent signals to rank candidates far beyond what keywords alone can achieve.

### Core Capabilities

| Capability | Description |
|---|---|
| **Semantic Understanding** | Dense vector embeddings capture context, synonyms & narrative coherence |
| **Verified Skill Trust** | Evidence-based competency scores using endorsements, tests & duration |
| **Behavioral Recruitability** | Real-time signals like activity recency, response rates & salary fit |
| **Fraud Defense** | Honeypot detection eliminates synthetic and spam profiles |
| **Scale Performance** | 100,000 candidates ranked in ~64 seconds on a single CPU |
| **Explainable Output** | Deterministic, human-readable reasoning for every shortlisted candidate |

---

## System Architecture

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


## Setup & Installation

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


## Output Format

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

## Performance Metrics

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

## Fraud & Quality Defense

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

## Technology Stack

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

## Running Tests

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


<div align="center">

---

*Built for the Redrob AI Challenge — India Runs Hackathon*

**Vettly** — *Find the signal. Ignore the noise.*

</div>
