import streamlit as st
import json
import time
import gzip
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.hard_filter import is_killed
from src.score_career import compute_A, compute_keyword_max
from src.score_skills import compute_B
from src.score_embed import compute_C_all
from src.availability import apply_multipliers
from src.output import write_submission
from src.precompute import build_candidate_text, build_jd_text

# Set Page Config
st.set_page_config(
    page_title="Vettly Talent Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Dark Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Backgrounds */
    .stApp {
        background-color: #0f111a;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #161925;
        border-right: 1px solid #24293e;
    }
    
    /* File Uploader Customization */
    [data-testid="stFileUploader"] {
        background-color: #1c2035;
        border: 1px dashed #24293e;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Slider & Accent styling */
    .stSlider > div > div > div > div {
        background-color: #ff7b00 !important;
    }
    
    /* Candidate Cards */
    .candidate-card {
        background-color: #161925;
        border: 1px solid #24293e;
        border-left: 4px solid #ff7b00;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        border-color: #ff7b00;
    }
    
    /* Badges & Metrics */
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        background-color: #24293e;
        color: #e2e8f0;
        border: 1px solid #333a56;
    }
    .badge-accent {
        color: #ff7b00;
        background-color: rgba(255, 123, 0, 0.1);
        border-color: rgba(255, 123, 0, 0.2);
    }
    
    /* Stats Layout */
    .stat-box {
        background-color: #161925;
        border: 1px solid #24293e;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .stat-box h4 {
        color: #94a3b8 !important;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .stat-box h2 {
        color: #ffffff !important;
        font-size: 2rem;
        margin: 0;
    }
    
    /* Typography adjustments */
    h1, h2, h3, h4, h5, p, span {
        font-family: 'Inter', sans-serif;
    }
    h1 { color: #ffffff !important; font-weight: 700; }
    h2, h3 { color: #e2e8f0 !important; font-weight: 600; }
    p { color: #cbd5e1 !important; }
    
    /* Override markdown text colors */
    .stMarkdown p { color: #cbd5e1; }
    .stMarkdown strong { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# App Title & Welcome Banner
st.markdown("<h1 style='font-size: 2.8rem; margin-bottom: 0;'>Vettly Talent Portal</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #94a3b8 !important; margin-top: 0.5rem; font-weight: 400;'>AI-Assisted Candidate Discovery, Fit Analysis, and Role Alignment</h3>", unsafe_allow_html=True)
st.markdown("<hr style='border-color: #24293e; margin: 2rem 0;'>", unsafe_allow_html=True)

# Center Uploads
st.markdown("### Upload Datasets")
col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded_jd = st.file_uploader("Job Description (JSON)", type=["json"])
with col_up2:
    uploaded_candidates = st.file_uploader("Candidates Dataset (JSONL or GZ)", type=["jsonl", "gz", "jsonl.gz"])
    use_default_candidates = st.checkbox("Use Demo Candidates Dataset (100,000 Profiles) - Instant Load", value=False)

if use_default_candidates:
    uploaded_candidates = open("data/candidates.jsonl", "rb")

# Sidebar Setup
with st.sidebar:
    st.markdown("### Score Weights Configuration")
    w_A = st.slider("Career Fit Weight (A)", 0.0, 1.0, 0.40, 0.05)
    w_B = st.slider("Skill Trust Weight (B)", 0.0, 1.0, 0.35, 0.05)
    w_C = st.slider("Semantic Similarity Weight (C)", 0.0, 1.0, 0.25, 0.05)
    
    if abs((w_A + w_B + w_C) - 1.0) > 0.001:
        st.warning(f"Weights sum to {w_A+w_B+w_C:.2f}. They will be normalized to 1.0 internally.")

# Check if inputs are uploaded
if not uploaded_jd or not uploaded_candidates:
    st.info("Welcome. Please upload the Job Description and the Candidates Dataset (or check the Demo Dataset box) above to begin. The start action will appear once files are loaded.")
else:
    jd_data = json.load(uploaded_jd)
    
    # Display JD summary info
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Job Specifications")
        st.write(f"**Target Role:** {jd_data.get('title', 'Unknown')}")
        st.write(f"**Required Experience:** {jd_data.get('min_yoe', 5)}+ years")
        st.write(f"**Max Budget:** {jd_data.get('budget_max_inr_lpa', 'N/A')} LPA")
        st.write(f"**Preferred Location(s):** {', '.join(jd_data.get('preferred_locations', []))}")

    with col2:
        st.markdown("### Focus Skills & Keywords")
        must_haves = jd_data.get("must_have_skills", [])
        st.markdown("**Must Have Skills:**")
        st.write(", ".join([f"`{s}`" for s in must_haves]))
        
        kws = jd_data.get("keywords", [])
        st.markdown("**Target Keywords:**")
        st.write(", ".join([f"`{k}`" for k in kws]))

    # Start Button
    if st.button("Start Talent Search & Vetting Pipeline", type="primary", use_container_width=True):
        # Normalize weights
        total_w = w_A + w_B + w_C
        nw_A, nw_B, nw_C = w_A/total_w, w_B/total_w, w_C/total_w
        
        status_box = st.empty()
        progress_bar = st.progress(0)
        
        # 1. Loading & Streaming from memory buffer
        status_box.info("Streaming uploaded candidates & fitting TF-IDF parameters...")
        progress_bar.progress(15)
        
        titles = []
        is_gz = getattr(uploaded_candidates, "name", "").endswith(".gz")
        
        def stream_file(fobj):
            fobj.seek(0)
            if is_gz:
                return gzip.GzipFile(fileobj=fobj, mode='rb')
            return fobj

        for line_bytes in stream_file(uploaded_candidates):
            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue
            cand = json.loads(line)
            title = cand.get("profile", {}).get("current_title") or cand.get("current_title") or ""
            titles.append(title)
            
        tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
        tfidf.fit(titles)
        del titles
        
        # 2. Hard Filtering
        status_box.info("Applying hard gatekeeper rules (Profile Completeness, Activity, Intent)...")
        progress_bar.progress(35)
        
        survivors = []
        killed_reasons = {}
        for line_bytes in stream_file(uploaded_candidates):
            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue
            cand = json.loads(line)
            killed_flag, reason = is_killed(cand, jd_data, tfidf)
            if killed_flag:
                # Categorize reason for display
                category = "Other Filter"
                if "profile_completeness_score" in reason:
                    category = "Incomplete Profile"
                elif "verified_email" in reason:
                    category = "Unverified Email"
                elif "interview_completion_rate" in reason:
                    category = "Low Interview Completion"
                elif "Inactive" in reason or "last_active_date" in reason:
                    category = "Inactive > 180 Days"
                elif "open_to_work_flag" in reason:
                    category = "Not Open to Work"
                elif "Zero industry overlap" in reason:
                    category = "Industry Mismatch"
                elif "Title similarity" in reason:
                    category = "Role/Title Mismatch"
                killed_reasons[category] = killed_reasons.get(category, 0) + 1
            else:
                survivors.append(cand)
                    
        # 3. Embedding Matching
        status_box.info(f"Generating semantic candidate vectors for {len(survivors)} surviving profiles...")
        progress_bar.progress(60)
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        jd_text = build_jd_text(jd_data)
        jd_vec = model.encode(jd_text, normalize_embeddings=True).astype("float32")
        
        survivor_texts = [build_candidate_text(s) for s in survivors]
        cand_vecs = model.encode(survivor_texts, batch_size=256, normalize_embeddings=True).astype("float32")
        
        # 4. Scoring
        status_box.info("Calculating comprehensive fit scores & multipliers...")
        progress_bar.progress(85)
        
        C_scores = compute_C_all(jd_vec, cand_vecs)
        C_map = {str(s.get("candidate_id") or s.get("id")): float(score) for s, score in zip(survivors, C_scores)}
        
        # Fit survivors TF-IDF
        tfidf_surv = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
        tfidf_surv.fit(survivor_texts)
        keyword_max = compute_keyword_max(survivors, jd_data, tfidf_surv)
        
        raw_scored = []
        for cand in survivors:
            cand_id = str(cand.get("candidate_id") or cand.get("id"))
            A_res = compute_A(cand, jd_data, tfidf_surv, keyword_max)
            B_res = compute_B(cand, jd_data)
            C = C_map.get(cand_id, 0.0)
            
            A = A_res["A"]
            B = B_res["B"]
            raw_score = round(nw_A * A + nw_B * B + nw_C * C, 4)
            
            raw_scored.append({
                "candidate_id": cand_id,
                "candidate": cand,
                "A": A,
                "B": B,
                "C": C,
                "raw_score": raw_score
            })
            
        final_scored = apply_multipliers(raw_scored, jd_data)
        
        # Sort and take Top 50 for display
        final_scored = sorted(final_scored, key=lambda x: x["final_score"], reverse=True)
        top_candidates = final_scored[:50]
        
        # Clear status
        status_box.empty()
        progress_bar.empty()
        
        # Display Stats Summary Dashboard
        st.markdown("<hr style='border-color: #24293e; margin: 2rem 0;'>", unsafe_allow_html=True)
        st.markdown("### Talent Pipeline Summary Dashboard")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.markdown(f"<div class='stat-box'><h4>Total Profiles</h4><h2>100,000</h2></div>", unsafe_allow_html=True)
        with d_col2:
            st.markdown(f"<div class='stat-box'><h4>Filtered Out</h4><h2>{100000 - len(survivors):,}</h2></div>", unsafe_allow_html=True)
        with d_col3:
            st.markdown(f"<div class='stat-box'><h4>Qualified Survivors</h4><h2 style='color: #ff7b00 !important;'>{len(survivors):,}</h2></div>", unsafe_allow_html=True)
        with d_col4:
            st.markdown(f"<div class='stat-box'><h4>Pruned Ratio</h4><h2>{((100000 - len(survivors))/100000)*100:.2f}%</h2></div>", unsafe_allow_html=True)
            
        # Draw Bar chart of filtering reasons
        st.markdown("<br><h4>Primary Reasons for Candidate Disqualification</h4>", unsafe_allow_html=True)
        df_reasons = pd.DataFrame(list(killed_reasons.items()), columns=["Disqualification Category", "Candidate Count"])
        st.bar_chart(df_reasons.set_index("Disqualification Category"), color="#ff7b00")
        
        st.markdown("<hr style='border-color: #24293e; margin: 2rem 0;'>", unsafe_allow_html=True)
        
        # Output Top Candidates list in a gorgeous card design
        col_title, col_export = st.columns([3, 1])
        with col_title:
            st.markdown("### Top 50 Matched Candidates")
        with col_export:
            # Prepare DataFrame for Excel export
            export_data = []
            for rank, cand_item in enumerate(top_candidates, 1):
                c = cand_item["candidate"]
                prof = c.get("profile", {})
                export_data.append({
                    "Rank": rank,
                    "Candidate ID": cand_item["candidate_id"],
                    "Name": prof.get("anonymized_name", "Anonymous"),
                    "Match %": int(cand_item["final_score"] * 100),
                    "Title": prof.get("current_title", "N/A"),
                    "Company": prof.get("current_company", "N/A"),
                    "Experience": prof.get("years_of_experience") or prof.get("yoe") or 0.0,
                    "Location": prof.get("location", "N/A")
                })
            df_export = pd.DataFrame(export_data)
            
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Top Candidates')
                
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name="vettly_top_candidates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        for rank, cand_item in enumerate(top_candidates, 1):
            cand = cand_item["candidate"]
            profile = cand.get("profile") or {}
            anom_name = profile.get("anonymized_name", "Anonymous Candidate")
            curr_title = profile.get("current_title", "Software Professional")
            curr_company = profile.get("current_company", "N/A")
            yoe = profile.get("years_of_experience") or profile.get("yoe") or 0.0
            loc = profile.get("location", "Remote")
            
            # Scores
            final_pct = int(cand_item["final_score"] * 100)
            score_A_pct = int(cand_item["A"] * 100)
            score_B_pct = int(cand_item["B"] * 100)
            score_C_pct = int(cand_item["C"] * 100)
            
            # HTML Card block
            st.markdown(f"""
            <div class="candidate-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                    <div>
                        <span class="metric-badge badge-accent" style="font-size:1rem; padding: 0.4rem 0.8rem;">Rank #{rank}</span>
                        <strong style="font-size:1.2rem; color:#ffffff; margin-left: 0.5rem;">{anom_name}</strong>
                        <span style="color:#94a3b8; margin-left:1rem;">{curr_title} @ {curr_company}</span>
                    </div>
                    <div>
                        <span style="font-size:1.6rem; font-weight:700; color:#ff7b00;">{final_pct}% Match</span>
                    </div>
                </div>
                <div style="margin-bottom: 0.4rem;">
                    <span class="metric-badge">Exp: {yoe} Yrs</span>
                    <span class="metric-badge">Loc: {loc}</span>
                    <span class="metric-badge">Career Fit: {score_A_pct}%</span>
                    <span class="metric-badge">Skills Trust: {score_B_pct}%</span>
                    <span class="metric-badge">Semantic Sim: {score_C_pct}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Details Expander
            with st.expander(f"Inspect Profile Details & Alignment: {anom_name}"):
                st.markdown("**Core Fit Analysis:**")
                st.write(f"Candidate has a match score of {final_pct}%. They possess {yoe} years of relevant industry experience in {profile.get('current_industry', 'tech')}. Matched locations include {loc}.")
                
                # Show career history
                st.markdown("**Career History Summary:**")
                for job in cand.get("career_history", []):
                    st.write(f"- **{job.get('title')}** at *{job.get('company')}* ({job.get('duration_months', 0)} months) — *{job.get('description', '')[:200]}...*")
                
                # Show skills
                st.markdown("**Technical Skills Inventory:**")
                skills_list = [s.get("name") if isinstance(s, dict) else s for s in cand.get("skills", [])]
                st.write(", ".join([f"`{s}`" for s in skills_list[:15]]))
